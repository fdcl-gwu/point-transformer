import logging
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .pointnet_util import DynamicPointNetSetAbstractionMsg, PointNetSetAbstractionMsg, PointNetSetAbstraction


#Mish - "Mish: A Self Regularized Non-Monotonic Neural Activation Function"
#https://arxiv.org/abs/1908.08681v1
#implemented for PyTorch / FastAI by lessw2020 
#github: https://github.com/lessw2020/mish

class Mish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x *( torch.tanh(F.softplus(x)))

def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.00)

def create_rFF(channel_list, input_dim):
    rFF = nn.ModuleList([nn.Conv2d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv2d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(input_dim,1)))

    return rFF

def create_rFF3d(channel_list, num_points, dim):
    rFF = nn.ModuleList([nn.Conv3d(in_channels=channel_list[i], 
                                   out_channels=channel_list[i+1],
                                   kernel_size=(1,1,1)) for i in range(len(channel_list) - 1)])
    rFF.insert(0, nn.Conv3d(in_channels=1, 
                            out_channels=channel_list[0], 
                            kernel_size=(1, num_points, dim)))

    return rFF


class PoseLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Pose loss for 6DOF estimation using:
        - Geodesic loss for rotation (axis-angle representation)
        - L2 loss for translation residual
        - Alpha scales the translation loss
        - Beta scales the rotation loss
        """
        super(PoseLoss, self).__init__()
        self.alpha = alpha  # Scaling factor for translation loss
        self.beta = beta # Scaling factor for rotation loss

    def forward(self, pred_r, gt_q, pred_t, gt_t):
        """
        Compute total pose loss:
        - Convert predicted axis-angle to rotation matrix.
        - Convert ground truth quaternion to rotation matrix.
        - Compute geodesic distance for rotation loss.
        - Compute L2 loss for translation residual.

        Inputs:
        - pred_r: Predicted rotation in axis-angle (B, 3)
        - gt_q: Ground truth rotation in quaternion (B, 4), wxyz format
        - pred_t: Predicted translation residual (B, 3)
        - gt_t: Ground truth translation (B, 3)

        Output:
        - Total loss: geodesic loss + weighted translation loss
        """
        # Convert ground truth quaternion to rotation matrix
        R_gt = self.quaternion_to_rotation_matrix(gt_q)

        # Convert predicted axis-angle to rotation matrix
        R_pred = self.axis_angle_to_rotation_matrix(pred_r)

        # Compute geodesic loss for rotation
        loss_r = self.geodesic_loss(R_pred, R_gt)

        # Compute L2 loss for translation (as before)
        loss_t = torch.linalg.vector_norm(pred_t - gt_t, ord=2, dim=-1).mean()

        # Weighted total loss
        total_loss = (self.alpha * loss_t) + (self.beta * loss_r)

        return total_loss

    def quaternion_to_rotation_matrix(self, q):
        """
        Converts a quaternion (B, 4) to a rotation matrix (B, 3, 3).
        """
        q = F.normalize(q, dim=-1)  # Ensure the quaternion is normalized
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        B = q.shape[0]
        R = torch.zeros((B, 3, 3), device=q.device)

        R[:, 0, 0] = 1 - 2 * (y**2 + z**2)
        R[:, 0, 1] = 2 * (x * y - w * z)
        R[:, 0, 2] = 2 * (x * z + w * y)
        R[:, 1, 0] = 2 * (x * y + w * z)
        R[:, 1, 1] = 1 - 2 * (x**2 + z**2)
        R[:, 1, 2] = 2 * (y * z - w * x)
        R[:, 2, 0] = 2 * (x * z - w * y)
        R[:, 2, 1] = 2 * (y * z + w * x)
        R[:, 2, 2] = 1 - 2 * (x**2 + y**2)

        return R

    def axis_angle_to_rotation_matrix(self, axis_angle):
        """
        Converts an axis-angle vector (B, 3) to a rotation matrix (B, 3, 3) using the exponential map.
        
        # Rodrigues' formula for rotation matrix with:

                    axis_angle = r
                    theta = ||r|| (2-norm)
                    axis = r / ||r||
        """
        theta = torch.linalg.vector_norm(axis_angle, dim=1, keepdim=True)
        eps = 1e-6
        axis = axis_angle / (theta + eps)
        theta = theta.squeeze(1)  # Shape [B]

        zeros = torch.zeros_like(theta)
        # K is already normalized from axis above, so no need to divide by theta like in the paper
        K = torch.stack([
            zeros, -axis[:, 2], axis[:, 1],
            axis[:, 2], zeros, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zeros
        ], dim=1).reshape(-1, 3, 3)

        I = torch.eye(3, device=axis.device).unsqueeze(0)
        sin_theta = torch.sin(theta).view(-1, 1, 1)  # (B, 1, 1)
        cos_theta = (1 - torch.cos(theta)).view(-1, 1, 1)  # (B, 1, 1)

        # Rodrigues' formula to rotation matrix
        R = I + sin_theta * K + cos_theta * torch.matmul(K, K)

        return R

    def geodesic_loss(self, R_pred, R_gt):
        """
        Computes the geodesic loss between two rotation matrices.
        Loss formula:
        L_r = arccos( (trace(R_pred * R_gt^T) - 1) / 2 )
        """

        # Compute trace of (R_pred * R_gt^T)
        trace_val = torch.einsum('bii->b', torch.matmul(R_pred, R_gt.transpose(1, 2))) #dim 0 is batch size

        # Clamp to avoid numerical errors leading to NaNs
        trace_val = torch.clamp((trace_val - 1) / 2, -1.0, 1.0)

        # Compute geodesic loss
        loss_r = torch.acos(trace_val)

        return loss_r.mean()

    

class KPLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()
        self.alpha = 1  # Class (cross entropy) scaling
        self.beta = 4    # Smooth L1 scaling for regression
        self.delta = 5 # rotation loss scaling
        self.epsilon = 6 # center loss scaling

    def forward(self, pred_keypoints, gt_keypoints, pred_section_logits, gt_section_label):
        """
        Parameters:
          pred_keypoints: Tensor [B, num_keypoints, 3]
          gt_keypoints: Tensor [B, num_keypoints, 3]
          pred_section_logits: Tensor [B, num_keypoints, num_sections]
          gt_section_label: Tensor [B, num_keypoints] (class indices)

        Returns:
          total_loss: scalar (combination of classification, keypoint regression, and structure constraints)
        """

        # 1. Keypoint Regression Loss (Smooth L1)
        keypoint_loss = F.smooth_l1_loss(pred_keypoints, gt_keypoints)

        # 2. Classification Loss for Keypoint Labels (Cross-Entropy)
        section_loss = F.cross_entropy(
            pred_section_logits.view(-1, pred_section_logits.size(-1)), 
            gt_section_label.view(-1)
        )

        # 3. Rotation Alignment Loss (Procrustes Analysis for Rotation Consistency)
        def rotation_loss(pred_keypoints, gt_keypoints):
            """Apply rotation loss per section."""
            loss = 0
            num_sections = pred_keypoints.shape[1] // 20
            for i in range(num_sections):
                pred = pred_keypoints[:, i * 20 : (i + 1) * 20, :3]
                gt = gt_keypoints[:, i * 20 : (i + 1) * 20, :3]
                pred_centered = pred - pred.mean(dim=1, keepdim=True)
                gt_centered = gt - gt.mean(dim=1, keepdim=True)
                U, _, V = torch.svd(torch.bmm(gt_centered.transpose(1, 2), pred_centered))
                R = torch.bmm(U, V.transpose(1, 2))
                pred_aligned = torch.bmm(pred_centered, R)
                loss += F.smooth_l1_loss(pred_aligned, gt_centered)
            return loss / num_sections

        rot_loss_val = rotation_loss(pred_keypoints, gt_keypoints)

        # 4. Center Alignment Loss (Ensures predicted centroids match ground truth)
        def center_loss(pred_keypoints, gt_keypoints):
            """Ensure predicted section centers align with GT centers."""
            loss = 0
            num_sections = pred_keypoints.shape[1] // 20
            for i in range(num_sections):
                pred_center = torch.mean(pred_keypoints[:, i * 20 : (i + 1) * 20, :3], dim=1)
                gt_center = torch.mean(gt_keypoints[:, i * 20 : (i + 1) * 20, :3], dim=1)
                loss += F.smooth_l1_loss(pred_center, gt_center)
            return loss / num_sections
        
        cent_loss_val = center_loss(pred_keypoints, gt_keypoints)

        # Final combined loss
        total_loss = (
            self.alpha * section_loss +
            self.beta * keypoint_loss +
            self.delta * rot_loss_val +
            self.epsilon * cent_loss_val
        )

        print(f"KPLoss: keypoint_loss: {keypoint_loss.item()}, section_loss: {section_loss.item()}, "
              f"rot_loss: {rot_loss_val.item()}, center_loss: {cent_loss_val.item()}, "
              f"total_loss: {total_loss.item()}")

        return total_loss
 
class Point_Transformer(nn.Module):
    def __init__(self, config):
        super(Point_Transformer, self).__init__()
        
        # Parameters
        self.actv_fn = Mish()

        self.p_dropout = config['dropout']
        self.norm_channel = config['use_labels']
        self.input_dim = 13 if config['use_labels'] else 3
        self.num_sort_nets = config['M']
        self.top_k = config['K']
        self.d_model = config['d_m']
        self.unit_sphere = config['unit_sphere']
 
        # TODO: try different radius values
        self.radius_max_points = config['radius_max_points']
        self.radius = config['radius']

        ## Create rFF to project input points to latent feature space
        ## Local Feature Generation --> rFF
        self.sort_ch = [64, 128]
        self.sort_cnn = create_rFF(self.sort_ch, self.input_dim)
        self.sort_cnn.apply(init_weights)
        self.sort_bn = nn.ModuleList([nn.BatchNorm2d(num_features=self.sort_ch[i]) for i in range(len(self.sort_ch))])
        
        ## Create Self-Attention layer
        ##  Local Feature Generation --> A^self
        self.input_selfattention_layer = nn.TransformerEncoderLayer(self.sort_ch[-1], nhead=8)

        self.sortnets = nn.ModuleList([SortNet(self.sort_ch[-1],
                                                  self.input_dim,
                                                  self.actv_fn,
                                                  top_k = self.top_k) for _ in range(self.num_sort_nets)])
     

        ## Create ball query search + feature aggregation of SortNet
        ## ball query + feat. agg
        ## Note: We put the ball query search and feature aggregation outside the SortNet implementation as it greatly decreased computational time
        ## This however, does not change the method in any way
        self.radius_ch = [128, 256, self.d_model-1-self.input_dim]
        self.radius_cnn = create_rFF3d(self.radius_ch, self.radius_max_points+1, self.input_dim)
        self.radius_cnn.apply(init_weights)
        self.radius_bn = nn.ModuleList([nn.BatchNorm3d(num_features=self.radius_ch[i]) for i in range(len(self.radius_ch))])

        ## Create set abstraction (MSG)
        ##  Global Feature Generation --> Set Abstraction (MSG)
        out_points = 128
        in_channel = 3 if self.norm_channel else 0 

        if self.unit_sphere:
            self.sa1 = PointNetSetAbstractionMsg(256, [0.1, 0.2, 0.4], [16, 32, 64], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = PointNetSetAbstractionMsg(out_points, [0.2, 0.4, 0.6], [32, 64, 128], 320,[[32, 64, 128], [64, 64, 128], [64, 128, 253]])
        else:
            self.sa1 = DynamicPointNetSetAbstractionMsg(256, [16, 32, 64], in_channel, [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
            self.sa2 = DynamicPointNetSetAbstractionMsg(out_points, [32, 64, 128], 320,[[32, 64, 128], [64, 64, 128], [64, 128, 253]])
    
        ## Create Local-Global Attention
        ##  A^LG
        out_dim = 64
        self.decoder_layer = nn.TransformerDecoderLayer(self.d_model, nhead=8)
        self.last_layer = PTransformerDecoderLayer(self.d_model, nhead=8, last_dim=out_dim)
        self.custom_decoder = PTransformerDecoder(self.decoder_layer, 1, self.last_layer)
        self.transformer_model = nn.Transformer(d_model=self.d_model,nhead=8, dim_feedforward=512, num_encoder_layers=1, num_decoder_layers=1, custom_decoder=self.custom_decoder)
        self.transformer_model.apply(init_weights)

        # Create the pose estimation heads
        dim_flatten = out_dim * self.num_sort_nets * self.top_k  # The global feature vector

        # Keypoint Prediction MLP (Outputs XYZ per keypoint)
        ## 06.03: HARDCODED FOR NOW
        self.num_keypoints = 40
        self.num_sections = 2

        self.keypoint_mlp = nn.Sequential(
            nn.Linear(dim_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keypoints * 3)  # [B, num_keypoints * 3]
        )

        # Section Classification MLP (Outputs logits per keypoint for classification)
        self.section_mlp = nn.Sequential(
            nn.Linear(dim_flatten, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_keypoints * self.num_sections)  # [B, num_keypoints * num_sections]
        )
        
        # Initialize weights
        self.keypoint_mlp.apply(init_weights)
        self.section_mlp.apply(init_weights)

    def forward(self, input, centroid, scale):

        #############################################
        ## Global Features 
        #############################################
        xyz = input
        # print("xyz shape PT:", xyz.shape)  # [B, C, N]
        B, _, _ = xyz.shape
        
        if self.norm_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        if self.unit_sphere:
            ## Set Abstraction with MSG
            l1_xyz, l1_points = self.sa1(xyz, norm)
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
            global_feat = torch.cat([l2_xyz, l2_points], dim=1)
        else:
            ## Compute dynamic radii per batch
            base_radii = [0.1, 0.2, 0.4]  # Original MSG radii
            adjusted_radii = torch.tensor(base_radii, device=scale.device, dtype=scale.dtype) * scale.view(-1, 1)
            print(f"adjusted_radii: {adjusted_radii}, and scale: {scale}")

            ## Set Abstraction with MSG
            print(f"xyz: {xyz.shape}")
            l1_xyz, l1_points = self.sa1(xyz, norm, adjusted_radii)
            print(f"l1_xyz: {l1_xyz.shape}, l1_points: {l1_points.shape}")
            l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, adjusted_radii)
            global_feat = torch.cat([l2_xyz, l2_points], dim=1)

        #############################################
        ## Local Features
        #############################################
        
        x_local = input.unsqueeze(dim=1)

        # Project to latent feature dim
        for i, sort_conv in enumerate(self.sort_cnn):
            bn = self.sort_bn[i]
            x_local = self.actv_fn(bn(sort_conv(x_local)))
        x_local = x_local.transpose(2,1)

        # Perform Self Attention
        x_local = x_local.squeeze(dim=1)
        x_local = x_local.permute(2,0,1)
        x_local = self.input_selfattention_layer(x_local)
        x_local = x_local.permute(1,2,0)
        x_local = x_local.unsqueeze(dim=1)
        # Concatenate outputs of SortNet
        # TODO: save these top K points for visualization
        x_local_sorted = torch.cat([sortnet(x_local, input)[0] for sortnet in self.sortnets], dim=-1)

        # this corresponds to s^j_i
        x_local_scores = x_local_sorted[: ,3:, :].permute(0,2,1)
        # this corresponds to p^j_i
        x_local_sorted = x_local_sorted[:, :3, :].permute(0,2,1)

        # Perform ball query search with feature aggregation
        all_points = input.squeeze(dim=1).permute(0,2,1)
        query_points = x_local_sorted
        radius_indices = query_ball_point(self.radius, self.radius_max_points,all_points[:,:,:3], query_points[:,:,:3])
        
        radius_points = index_points(all_points, radius_indices) 
        radius_centroids = query_points.unsqueeze(dim=-2)

        # This corresponds to g^j
        # print("radius_centroids shape:", radius_centroids.shape)  # Expected [B, ?, 4]
        # print("radius_points shape before fix:", radius_points.shape)  # Expected [B, ?, 3]
        radius_grouped = torch.cat([radius_centroids, radius_points], dim=-2).unsqueeze(dim=1)

        for i, radius_conv in enumerate(self.radius_cnn):
            bn = self.radius_bn[i]
            radius_grouped = self.actv_fn(bn(radius_conv(radius_grouped)))

        radius_grouped = radius_grouped.squeeze()
        # This corresponds to f^j_i
        radius_grouped = torch.cat([x_local_sorted.transpose(2,1), radius_grouped, x_local_scores.transpose(2,1)], dim=1)

        #############################################
        ## Point Transformer
        #############################################

        source = global_feat.permute(2,0,1)
        target = radius_grouped.permute(2,0,1)

        embedding = self.transformer_model(source, target)
        embedding = embedding.permute(1, 2, 0)

        #############################################
        ## Keypoint Prediction
        #############################################

        # Flatten the feature vector for MLP heads
        global_features = torch.flatten(embedding, start_dim=1)  # [B, dim_flatten]

        # Predict keypoints
        pred_keypoints = self.keypoint_mlp(global_features)  # [B, num_keypoints * 3]
        pred_keypoints = pred_keypoints.view(-1, self.num_keypoints, 3)  # Reshape to [B, num_keypoints, 3]
        # scale keypoints back to original scale
        # print("Predicted keypoints shape:", pred_keypoints.shape)  # Expect [B, 40, 3]
        # print("Scale shape after view:", scale.view(B, 1, 1).shape)  # Expect [B, 1, 1]
        # print("Centroid shape after view:", centroid.view(B, 1, 3).shape)  # Expect [B, 1, 3]
        # pred_keypoints = pred_keypoints * scale.view(B, 1, 1) + centroid.view(B, 1, 3)
 
        # Predict keypoint region labels (class scores)
        pred_section_logits = self.section_mlp(global_features)  # [B, num_keypoints * num_sections]
        pred_section_logits = pred_section_logits.view(-1, self.num_keypoints, self.num_sections)

        return pred_keypoints, pred_section_logits


class SortNet(nn.Module):
    def __init__(self, num_feat, input_dims, actv_fn=F.relu, top_k = 5):
        super(SortNet, self).__init__()

        self.num_feat = num_feat
        self.actv_fn = actv_fn
        self.input_dims = input_dims

        self.top_k = top_k

        self.feat_channels =  [64, 16, 1]
        self.feat_generator = create_rFF(self.feat_channels, num_feat)
        self.feat_generator.apply(init_weights)
        self.feat_bn = nn.ModuleList([nn.BatchNorm2d(num_features=self.feat_channels[i]) for i in range(len(self.feat_channels))])

    def forward(self, sortvec, input):
        top_k = self.top_k
        batch_size = input.shape[0]
        feat_dim = input.shape[1]

        for i, conv in enumerate(self.feat_generator):
            bn = self.feat_bn[i]
            sortvec = self.actv_fn(bn(conv(sortvec)))
        sortvec = sortvec.squeeze(dim=1)


        topk = torch.topk(sortvec, k=top_k, dim=-1)
        indices = topk.indices.squeeze()
        sorted_input = index_points(input.permute(0,2,1), indices).permute(0,2,1)
        sorted_score = index_points(sortvec.permute(0,2,1), indices).permute(0,2,1)

        feat = torch.cat([sorted_input, sorted_score], dim=1)      

        return feat, indices

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class PTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, last_dim=64, dropout=0.1, activation=F.relu):
        super(PTransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, 256)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(256, last_dim)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
        
        # print("Unused kwargs PTransformerDecoderLayer:", kwargs)

        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        
        tgt = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
       
        return tgt


class PTransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers

    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """

    def __init__(self, decoder_layer, num_layers, last_layer, norm=None):
        super(PTransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.last_layer = last_layer
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None, **kwargs):
        
        # print("Unused kwargs PTransformerDecoder:", kwargs)

        r"""Pass the inputs (and mask) through the decoder layer in turn.

        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequnce from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for i in range(self.num_layers):
            output = self.layers[i](output, memory, tgt_mask=tgt_mask,
                                    memory_mask=memory_mask,
                                    tgt_key_padding_mask=tgt_key_padding_mask,
                                    memory_key_padding_mask=memory_key_padding_mask)

        if self.norm:
            output = self.norm(output)
            
        output = self.last_layer(output, memory)

        return output

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm?
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    torch.cuda.empty_cache()
    new_xyz = index_points(xyz, fps_idx)
    torch.cuda.empty_cache()
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points
