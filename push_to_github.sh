#!/bin/bash
# POT-SAM2 Hybrid GitHub Push Script

echo "========================================="
echo "POT-SAM2 Hybrid GitHub Push Script"
echo "========================================="
echo ""

# Check if remote already exists
if git remote | grep -q origin; then
    echo "⚠️  Remote 'origin' already exists."
    echo "Current remote URL:"
    git remote get-url origin
    echo ""
    read -p "Do you want to update it? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git remote set-url origin https://github.com/suraimukun777/POT-SAM2-Hybrid.git
        echo "✅ Remote URL updated"
    fi
else
    echo "Adding remote repository..."
    git remote add origin https://github.com/suraimukun777/POT-SAM2-Hybrid.git
    echo "✅ Remote added"
fi

echo ""
echo "Current branch:"
git branch --show-current

echo ""
echo "Switching to 'main' branch..."
git branch -M main
echo "✅ Branch renamed to 'main'"

echo ""
echo "========================================="
echo "Ready to push to GitHub!"
echo "========================================="
echo ""
echo "Repository: https://github.com/suraimukun777/POT-SAM2-Hybrid"
echo ""
read -p "Do you want to push now? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Pushing to GitHub..."
    git push -u origin main
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "========================================="
        echo "✅ Successfully pushed to GitHub!"
        echo "========================================="
        echo ""
        echo "View your repository at:"
        echo "https://github.com/suraimukun777/POT-SAM2-Hybrid"
        echo ""
    else
        echo ""
        echo "========================================="
        echo "❌ Push failed!"
        echo "========================================="
        echo ""
        echo "Possible reasons:"
        echo "1. Repository doesn't exist on GitHub yet"
        echo "2. Authentication issues"
        echo ""
        echo "Please:"
        echo "1. Create repository at: https://github.com/new"
        echo "   Name: POT-SAM2-Hybrid"
        echo "2. Make sure you're authenticated with GitHub"
        echo ""
    fi
else
    echo ""
    echo "Push cancelled. Run this script again when ready."
    echo ""
fi

