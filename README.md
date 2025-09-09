# 🎯 S-MoE Integration in U-Net Architecture

## 📋 **1. Overall S-MoE U-Net Architecture**

```
🏗️ STANDARD U-NET STRUCTURE WITH S-MoE INTEGRATION:

Input (Complex Spec) [batch, 1, 257, 188]
         ↓
    Global Router ← Makes ONE routing decision for entire network
         ↓
    Input Conv [1 → 45 channels]
         ↓
┌─── Encoder Path ───┐           ┌─── Decoder Path ───┐
│                    │           │                    │
│ EncoderBlock 1     │           │ DecoderBlock 4     │
│ [45 → 90]          │     ┌─────│ [180 → 90] + Skip │
│        ↓           │     │     │        ↓           │
│ EncoderBlock 2     │     │     │ S-MoE Layer ← Optional
│ [90 → 180] + S-MoE │ ────┼─────│        ↓           │
│        ↓           │     │     │ DecoderBlock 3     │
│ EncoderBlock 3     │     │     │ [90 → 45] + Skip   │
│ [180 → 360]        │ ────┼─────│        ↓           │
│        ↓           │     │     │ DecoderBlock 2     │
│ Bottleneck + S-MoE │     │     │ [45 → 45] + Skip   │
└────────────────────┘     │     └────────────────────┘
                           │              ↓
                      Skip Connections   Output Conv
                                         [45 → 1]
                                              ↓
                                    Enhanced Mask [batch, 1, 257, 188]
```

## 🧠 **2. Global Routing Strategy**

### **Key Innovation: ONE Router for ALL S-MoE Layers**

```python
class SMoEEnhancedUNet(nn.Module):
    def __init__(self, config):
        # ✅ SINGLE GLOBAL ROUTER - Makes decisions for entire network
        self.global_router = GlobalDistortionRouter(config)
        
        # All S-MoE layers will use the same routing decision
        # This ensures consistency across the entire network
        
    def forward(self, x, distortion_labels=None):
        # 1. MAKE SINGLE ROUTING DECISION
        routing_decision = self.global_router(x, distortion_labels)
        # Returns: {'expert_weights': [batch, n_experts], 'confidence': [batch, 1]}
        
        # 2. ALL S-MoE LAYERS USE THE SAME ROUTING DECISION
        for layer in self.encoder_layers:
            if 'smoe' in layer:
                features = layer['smoe'](features, routing_decision)  # Same routing!
                
        if 'smoe' in self.bottleneck:
            features = self.bottleneck['smoe'](features, routing_decision)  # Same routing!
            
        for layer in self.decoder_layers:
            if 'smoe' in layer:
                features = layer['smoe'](features, routing_decision)  # Same routing!
```

**Why Global Routing?**
- **Consistency**: Same distortion → same experts across all layers
- **Efficiency**: Only one routing computation per sample
- **Interpretability**: Clear understanding of which experts are active

## 🎪 **3. Strategic S-MoE Placement in U-Net**

### **Placement Strategy:**

```python
def _build_encoder(self):
    encoder_layers = nn.ModuleList()
    
    for i in range(self.n_layers):  # e.g., 5 layers for DCUNet-20
        layer_dict = nn.ModuleDict()
        
        # ALWAYS: Standard U-Net processing
        layer_dict['encoder_block'] = EncoderBlock(in_ch, out_ch)
        
        # CONDITIONAL: Add S-MoE at strategic positions
        if i == 1 and 'early_encoder' in self.config.smoe_layer_positions:
            layer_dict['smoe'] = SMoELayer(out_ch, 'early_encoder', self.config)
        elif i == 2 and 'mid_encoder' in self.config.smoe_layer_positions:
            layer_dict['smoe'] = SMoELayer(out_ch, 'mid_encoder', self.config)
            
        encoder_layers.append(layer_dict)
    
    return encoder_layers
```

### **Default Placement Configuration:**

```python
smoe_layer_positions = [
    'early_encoder',   # Layer 1: After 45→90 expansion  
    'mid_encoder',     # Layer 2: After 90→180 expansion
    'bottleneck',      # Bottleneck: Highest semantic level
    'mid_decoder',     # Decoder 2: During 180→90 reconstruction  
    'late_decoder'     # Decoder 1: During 90→45 final refinement
]
```

## 🔄 **4. Detailed Forward Pass Flow**

### **Complete Forward Pass with S-MoE:**

```python
def forward(self, x, distortion_labels=None):
    # INPUT: Complex spectrogram [batch, 1, 257, 188]
    
    # STEP 1: GLOBAL ROUTING DECISION
    routing_decision = self.global_router(x, distortion_labels)
    # Output: {'expert_weights': [batch, 2], 'confidence': [batch, 1]}
    # For 2-expert: expert_weights = [[0.8, 0.2], [0.1, 0.9], ...]
    #               meaning [noise_weight, reverb_weight] per sample
    
    # STEP 2: INPUT PROCESSING
    current = self.input_conv(x)  # [batch, 1, 257, 188] → [batch, 45, 257, 188]
    skip_connections = []
    
    # STEP 3: ENCODER PATH WITH S-MoE
    for i, layer_dict in enumerate(self.encoder_layers):
        print(f"Encoder Layer {i}:")
        
        # Standard U-Net processing
        skip, current = layer_dict['encoder_block'](current)
        print(f"  After EncoderBlock: skip={skip.shape}, current={current.shape}")
        
        # S-MoE processing (if present)
        if 'smoe' in layer_dict:
            print(f"  Applying S-MoE at position: {layer_dict['smoe'].layer_position}")
            skip = layer_dict['smoe'](skip, routing_decision)  # ← SAME ROUTING!
            print(f"  After S-MoE: skip={skip.shape}")
        
        skip_connections.append(skip)
        
    # Example output:
    # Encoder Layer 0: skip=[batch, 45, 257, 188], current=[batch, 45, 128, 94]
    # Encoder Layer 1: skip=[batch, 90, 128, 94], current=[batch, 90, 64, 47]
    #   Applying S-MoE at position: early_encoder
    #   After S-MoE: skip=[batch, 90, 128, 94]
    # Encoder Layer 2: skip=[batch, 180, 64, 47], current=[batch, 180, 32, 23]
    #   Applying S-MoE at position: mid_encoder  
    #   After S-MoE: skip=[batch, 180, 64, 47]
    
    # STEP 4: BOTTLENECK WITH S-MoE
    current = self.bottleneck['bottleneck_block'](current)
    if 'smoe' in self.bottleneck:
        print(f"Applying S-MoE at bottleneck")
        current = self.bottleneck['smoe'](current, routing_decision)  # ← SAME ROUTING!
    
    # STEP 5: DECODER PATH WITH S-MoE  
    for i, layer_dict in enumerate(self.decoder_layers):
        skip = skip_connections[-(i + 1)]  # Reverse order
        
        # Standard U-Net processing
        current = layer_dict['decoder_block'](current, skip)
        
        # S-MoE processing (if present)
        if 'smoe' in layer_dict:
            current = layer_dict['smoe'](current, routing_decision)  # ← SAME ROUTING!
    
    # STEP 6: OUTPUT
    mask = self.output_conv(current)  # [batch, 45, 257, 188] → [batch, 1, 257, 188]
    enhanced_spec = mask * x
    
    return {
        'enhanced_spec': enhanced_spec,
        'mask': mask,
        'routing_decision': routing_decision,
        'expert_usage': self._compute_expert_usage(routing_decision)
    }
```

## 🎯 **5. S-MoE Layer Implementation**

### **Individual S-MoE Layer Structure:**

```python
class SMoELayer(nn.Module):
    def __init__(self, input_dim, layer_position, config):
        # Each S-MoE layer contains multiple experts
        self.experts = nn.ModuleList([
            DistortionSpecificExpert(input_dim, distortion_type, layer_position, config)
            for distortion_type in range(config.n_experts)
        ])
        
        # For 2-expert system:
        # experts[0] = NoiseExpert
        # experts[1] = ReverbExpert
        
    def forward(self, x, routing_decision):
        # INPUT: x = [batch, channels, height, width]
        # INPUT: routing_decision['expert_weights'] = [batch, n_experts]
        
        residual = x  # Store for residual connection
        
        expert_weights = routing_decision['expert_weights']  # [batch, 2]
        batch_size = x.shape[0]
        
        if self.config.enable_multi_expert:
            # MULTI-EXPERT MODE: Weighted combination
            expert_outputs = []
            for batch_idx in range(batch_size):
                sample_input = x[batch_idx:batch_idx+1]
                weights = expert_weights[batch_idx]  # [2] - [noise_weight, reverb_weight]
                
                # Process with both experts
                noise_output = self.experts[0](sample_input)   # NoiseExpert
                reverb_output = self.experts[1](sample_input)  # ReverbExpert
                
                # Weighted combination
                combined = weights[0] * noise_output + weights[1] * reverb_output
                expert_outputs.append(combined)
            
            output = torch.cat(expert_outputs, dim=0)
        
        else:
            # SINGLE-EXPERT MODE: Choose best expert per sample
            expert_indices = torch.argmax(expert_weights, dim=-1)  # [batch]
            
            expert_outputs = []
            for batch_idx in range(batch_size):
                sample_input = x[batch_idx:batch_idx+1]
                expert_idx = expert_indices[batch_idx].item()
                
                expert_output = self.experts[expert_idx](sample_input)
                expert_outputs.append(expert_output)
            
            output = torch.cat(expert_outputs, dim=0)
        
        # RESIDUAL CONNECTION: Very important!
        return residual + output
```

## 🧪 **6. Two-Expert System Specific Implementation**

### **Simplified 2-Expert Router:**

```python
class TwoExpertRouter(nn.Module):
    def forward(self, complex_spec, distortion_labels=None):
        batch_size = complex_spec.shape[0]
        
        if self.training and distortion_labels is not None:
            # SUPERVISED ROUTING: Use ground truth labels
            expert_weights = torch.zeros(batch_size, 2)  # [batch, 2]
            
            for i in range(batch_size):
                label = distortion_labels[i].item()
                if label == 0:  # NOISE_ONLY
                    expert_weights[i] = torch.tensor([1.0, 0.0])  # Only noise expert
                elif label == 1:  # REVERB_ONLY  
                    expert_weights[i] = torch.tensor([0.0, 1.0])  # Only reverb expert
                elif label == 2:  # BOTH
                    expert_weights[i] = torch.tensor([0.5, 0.5])  # Both experts equally
        
        else:
            # IMPLICIT ROUTING: Learn from audio features
            features = self.extract_features(complex_spec)  # [batch, feature_dim]
            routing_logits = self.routing_network(features)  # [batch, 3]
            routing_probs = F.softmax(routing_logits, dim=-1)  # [batch, 3]
            
            # Convert to expert weights
            expert_weights = torch.zeros(batch_size, 2)
            expert_weights[:, 0] = routing_probs[:, 0] + routing_probs[:, 2]  # noise
            expert_weights[:, 1] = routing_probs[:, 1] + routing_probs[:, 2]  # reverb
        
        return {
            'expert_weights': expert_weights,  # [batch, 2]
            'confidence': self.compute_confidence(...)
        }
```

### **2-Expert MoE Layer:**

```python
class TwoExpertMoELayer(nn.Module):
    def __init__(self, input_dim, layer_position, config):
        # Only 2 experts: Noise + Reverb
        self.noise_expert = TwoExpertSpecialist(input_dim, 'noise', layer_position, config)
        self.reverb_expert = TwoExpertSpecialist(input_dim, 'reverb', layer_position, config)
        
    def forward(self, x, routing_decision):
        residual = x
        expert_weights = routing_decision['expert_weights']  # [batch, 2]
        
        # Process with both experts
        noise_output = self.noise_expert(x)    # [batch, channels, h, w]
        reverb_output = self.reverb_expert(x)  # [batch, channels, h, w]
        
        # Weighted combination
        noise_weights = expert_weights[:, 0].view(-1, 1, 1, 1)   # [batch, 1, 1, 1] 
        reverb_weights = expert_weights[:, 1].view(-1, 1, 1, 1)  # [batch, 1, 1, 1]
        
        combined_output = noise_weights * noise_output + reverb_weights * reverb_output
        
        # Residual connection
        return residual + combined_output
```

## 📊 **7. Expert Specialization by Layer Position**

### **Layer-Aware Expert Architecture:**

```python
class TwoExpertSpecialist(nn.Module):
    def _build_expert_network(self, config):
        if self.layer_position == 'bottleneck':
            # BOTTLENECK: Smaller networks due to high dimensionality
            hidden_dim = min(self.hidden_dim, 512)
            
            if self.expert_type == 'noise':
                return nn.Sequential(
                    ComplexConv2d(self.input_dim, hidden_dim, kernel_size=3, padding=1),
                    ComplexActivation('crelu'),
                    ComplexConv2d(hidden_dim, self.input_dim, kernel_size=1)
                )
            elif self.expert_type == 'reverb':
                return nn.Sequential(
                    ComplexConv2d(self.input_dim, hidden_dim, kernel_size=(3, 5), padding=(1, 2)),
                    ComplexActivation('crelu'),
                    ComplexConv2d(hidden_dim, self.input_dim, kernel_size=1)
                )
        
        else:
            # ENCODER/DECODER: Full expert networks
            if self.expert_type == 'noise':
                # NOISE EXPERT: Point-wise and local spatial processing
                return nn.Sequential(
                    ComplexConv2d(self.input_dim, self.hidden_dim, kernel_size=3, padding=1),
                    ComplexActivation('crelu'),
                    ComplexConv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                    ComplexActivation('crelu'),
                    ComplexConv2d(self.hidden_dim, self.input_dim, kernel_size=1)
                )
            elif self.expert_type == 'reverb':
                # REVERB EXPERT: Temporal processing with larger kernels
                return nn.Sequential(
                    ComplexConv2d(self.input_dim, self.hidden_dim, kernel_size=(3, 7), padding=(1, 3)),
                    ComplexActivation('crelu'),
                    ComplexConv2d(self.hidden_dim, self.hidden_dim, kernel_size=(3, 5), padding=(1, 2)),
                    ComplexActivation('crelu'),
                    ComplexConv2d(self.hidden_dim, self.input_dim, kernel_size=1)
                )
```

## 🎯 **8. Key Implementation Benefits**

### **1. Consistency Through Global Routing**
- ONE routing decision used everywhere
- No conflicts between different S-MoE layers
- Clear interpretation: "This sample uses 80% noise expert, 20% reverb expert"

### **2. Strategic Placement**
- **Early Encoder**: Pattern recognition level
- **Mid Encoder**: Feature extraction level  
- **Bottleneck**: High-level semantic processing
- **Mid/Late Decoder**: Reconstruction refinement

### **3. Residual Connections**
- S-MoE layers enhance features rather than replace them
- `output = input + expert_processing(input)`
- Prevents gradient vanishing in deep networks

### **4. Expert Specialization**
- **Noise Expert**: 3×3 kernels for local denoising
- **Reverb Expert**: (3×7) and (3×5) kernels for temporal processing
- **Layer-aware**: Smaller networks at bottleneck, full networks elsewhere

## 🔄 **9. Training Flow**

```python
# During training, the complete flow:
for batch in train_loader:
    clean, noisy, labels = batch['clean'], batch['noisy'], batch['distortion_label']
    
    # Forward pass with supervision
    outputs = model(noisy, labels)  # ← Labels provide routing supervision
    
    # The labels guide the router:
    # labels=[0, 1, 2] means [noise_only, reverb_only, both]
    # Router learns: label=0 → expert_weights=[1.0, 0.0]
    #               label=1 → expert_weights=[0.0, 1.0]  
    #               label=2 → expert_weights=[0.5, 0.5]
    
    enhanced = outputs['enhanced_waveform']
    routing_info = outputs['routing_decision']
    
    # Loss computation
    enhancement_loss = loss_fn(enhanced, clean, noisy)
    
    # Optional: Add routing regularization
    # routing_loss = routing_regularization(routing_info, labels)
    # total_loss = enhancement_loss + 0.1 * routing_loss
    
    enhancement_loss.backward()
    optimizer.step()
```

This implementation provides **interpretable**, **consistent**, and **efficient** S-MoE integration into the U-Net architecture! 🚀