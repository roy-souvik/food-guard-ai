#!/bin/bash
# Setup script for installing dependencies with ROCM PyTorch support

echo "🚀 Installing FoodGuard dependencies with ROCM PyTorch..."
echo ""

# Step 1: Uninstall existing torch packages
echo "1️⃣  Removing existing PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Step 2: Install ROCM PyTorch packages
echo "2️⃣  Installing PyTorch packages from ROCM 6.4 index..."
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.4

# Step 3: Install remaining dependencies
echo "3️⃣  Installing remaining dependencies from PyPI..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo "📝 To verify ROCM support, run cell 3 in the notebook to check torch.version.hip"
