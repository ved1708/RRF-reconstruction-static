# GitHub Setup Instructions

## Step 1: Create GitHub Repository

1. Go to https://github.com/new
2. Fill in repository details:
   - **Repository name**: `RRF-Reconstruction-Pipeline` (or your preferred name)
   - **Description**: `Complete end-to-end pipeline for reconstructing Radio-Frequency Radiance Fields using 3D Gaussian Splatting`
   - **Visibility**: Public (recommended) or Private
   - **Do NOT initialize with README** (we already have one)
3. Click "Create repository"

## Step 2: Link Local Repository to GitHub

After creating the repository on GitHub, you'll see a page with commands. Use these:

```bash
cd /home/ved/Ved/Project_1

# Add GitHub remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline.git

# OR use SSH if you have SSH keys set up:
# git remote add origin git@github.com:YOUR_USERNAME/RRF-Reconstruction-Pipeline.git

# Verify remote
git remote -v

# Push to GitHub
git push -u origin main
```

## Step 3: Verify Upload

Go to your repository URL:
```
https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline
```

You should see:
- ✅ README.md displayed on the home page
- ✅ All Python scripts
- ✅ Documentation in `docs/` folder
- ✅ requirements.txt
- ✅ .gitignore (preventing large files from being tracked)

## Step 4: Add Repository Topics (Optional)

On your GitHub repository page:
1. Click ⚙️ (Settings gear) next to "About"
2. Add topics:
   - `3d-gaussian-splatting`
   - `radio-frequency`
   - `radiance-fields`
   - `sionna`
   - `rf-propagation`
   - `indoor-localization`
   - `pytorch`
   - `tensorflow`
   - `blender`
   - `computer-graphics`
   - `wireless-communication`

## Step 5: Enable GitHub Pages (Optional)

To create a website for your documentation:

1. Go to repository Settings → Pages
2. Under "Source", select "Deploy from a branch"
3. Select branch: `main`, folder: `/ (root)`
4. Click "Save"
5. Your documentation will be available at:
   ```
   https://YOUR_USERNAME.github.io/RRF-Reconstruction-Pipeline/
   ```

## Step 6: Add README Badges (Optional)

Add these badges to the top of your README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange.svg)
![CUDA](https://img.shields.io/badge/CUDA-11.8+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
```

## Step 7: Create a Release (When Ready)

After testing and verifying everything works:

1. Go to repository → Releases → "Create a new release"
2. Tag version: `v1.0.0`
3. Release title: `Initial Release - Complete RRF Pipeline`
4. Description:
   ```markdown
   ## 🎉 First Stable Release
   
   Complete pipeline for Radio-Frequency Radiance Field reconstruction:
   - Parametric 3D scene generation
   - Visual dataset generation with Blender
   - RF dataset generation with Sionna RT
   - Two-stage 3DGS training
   - Comprehensive evaluation tools
   - Interactive 3D visualization
   
   ### Features
   - ✅ Full documentation for every step
   - ✅ Working examples on custom 7×5×3m room
   - ✅ Localization accuracy: ~0.5m RMSE
   - ✅ Visual quality: 30+ dB PSNR
   - ✅ RF reconstruction: 27+ dB PSNR
   
   ### Requirements
   - NVIDIA GPU with CUDA 11.8+
   - 16GB+ RAM
   - Ubuntu 20.04+ or similar
   
   See [Installation Guide](README.md#installation) for setup instructions.
   ```

## Troubleshooting

### Authentication Issues

If you get authentication errors:

**Option 1: HTTPS with Personal Access Token**
```bash
# Generate token: GitHub Settings → Developer settings → Personal access tokens → Generate new token
# Use token as password when prompted

git push -u origin main
# Username: YOUR_USERNAME
# Password: <paste your token>
```

**Option 2: SSH Keys** (recommended)
```bash
# Generate SSH key
ssh-keygen -t ed25519 -C "your_email@example.com"

# Copy public key
cat ~/.ssh/id_ed25519.pub

# Add to GitHub: Settings → SSH and GPG keys → New SSH key
# Then use SSH URL:
git remote set-url origin git@github.com:YOUR_USERNAME/RRF-Reconstruction-Pipeline.git
git push -u origin main
```

### Large Files Warning

If you see warnings about large files:

```bash
# Check file sizes
find . -type f -size +50M

# If needed, add to .gitignore and remove from tracking
echo "path/to/large/file" >> .gitignore
git rm --cached path/to/large/file
git commit -m "Remove large files"
```

### RF-3DGS Submodule

The RF-3DGS folder is intentionally excluded (it's a separate repository). Users should clone it separately:

```bash
cd Project_1
git clone https://github.com/Wangmz-1203/RF-3DGS.git
```

Or add it as a proper submodule:
```bash
git submodule add https://github.com/Wangmz-1203/RF-3DGS.git RF-3DGS
git commit -m "Add RF-3DGS as submodule"
git push
```

## What Gets Pushed

✅ **Included**:
- All Python scripts
- Documentation (README.md, docs/*.md)
- Configuration files (requirements.txt, .gitignore)
- Scene XML files
- Jupyter notebooks

❌ **Excluded** (via .gitignore):
- Large datasets (dataset_visual_v2/, dataset_custom_scene_ideal_mpc/)
- Output models (output/)
- PLY files (room_5x3x3_combined.ply)
- Pickle files (rf_dataset.pkl)
- Result images and videos

## Next Steps

After pushing to GitHub:

1. **Add LICENSE**: Choose MIT, Apache 2.0, or BSD 3-Clause
2. **Add CITATION.cff**: For academic citations
3. **Create Issues**: Track bugs and feature requests
4. **Enable Discussions**: Community Q&A
5. **Add Contributors**: If working with a team
6. **Write blog post**: Share your work!

## Example README Header

Update your README.md to include GitHub repository info:

```markdown
# Radio-Frequency Radiance Fields (RRF) Reconstruction Pipeline

[![GitHub stars](https://img.shields.io/github/stars/YOUR_USERNAME/RRF-Reconstruction-Pipeline)](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/YOUR_USERNAME/RRF-Reconstruction-Pipeline)](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/network)
[![GitHub issues](https://img.shields.io/github/issues/YOUR_USERNAME/RRF-Reconstruction-Pipeline)](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/issues)

Complete end-to-end pipeline for reconstructing Radio-Frequency Radiance Fields from 3D scenes using 3D Gaussian Splatting.

[Documentation](https://github.com/YOUR_USERNAME/RRF-Reconstruction-Pipeline/tree/main/docs) | [Installation](#installation) | [Quick Start](#quick-start) | [Results](#results)
```

---

**Ready to push!** Follow the commands in Step 2 to upload your work to GitHub.
