#!/usr/bin/env python3
"""
find_large_files.py - Find files that would exceed GitHub's size limits

Run this BEFORE pushing to GitHub to identify problematic files.

Usage:
    python scripts/find_large_files.py [directory] [--limit MB]
    
Example:
    python scripts/find_large_files.py . --limit 50
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

# GitHub limits
GITHUB_FILE_LIMIT_MB = 100  # Hard limit
GITHUB_WARNING_MB = 50      # Warning threshold
GITHUB_REPO_LIMIT_GB = 1    # Recommended max repo size

def get_size_mb(path):
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)

def get_size_gb(size_bytes):
    """Convert bytes to GB."""
    return size_bytes / (1024 * 1024 * 1024)

def find_large_files(directory, limit_mb=50):
    """Find all files larger than limit_mb."""
    large_files = []
    total_size = 0
    file_types = defaultdict(lambda: {'count': 0, 'size': 0})
    
    for root, dirs, files in os.walk(directory):
        # Skip hidden directories and common large directories
        dirs[:] = [d for d in dirs if not d.startswith('.') 
                   and d not in ['venv', 'env', 'node_modules', '__pycache__', 
                                 '.git', 'wandb', 'checkpoints', 'data']]
        
        for file in files:
            filepath = os.path.join(root, file)
            try:
                size_mb = get_size_mb(filepath)
                total_size += size_mb
                
                # Track by extension
                ext = Path(file).suffix.lower() or 'no_extension'
                file_types[ext]['count'] += 1
                file_types[ext]['size'] += size_mb
                
                if size_mb >= limit_mb:
                    large_files.append({
                        'path': filepath,
                        'size_mb': size_mb,
                        'extension': ext
                    })
            except (OSError, PermissionError):
                continue
    
    return large_files, total_size, file_types

def main():
    parser = argparse.ArgumentParser(description='Find large files for GitHub')
    parser.add_argument('directory', nargs='?', default='.', 
                        help='Directory to scan (default: current)')
    parser.add_argument('--limit', type=float, default=50,
                        help='Size limit in MB (default: 50)')
    parser.add_argument('--all', action='store_true',
                        help='Include hidden directories')
    args = parser.parse_args()
    
    print("=" * 60)
    print("GitHub Size Limit Checker")
    print("=" * 60)
    print(f"\nScanning: {os.path.abspath(args.directory)}")
    print(f"Threshold: {args.limit} MB\n")
    
    large_files, total_size, file_types = find_large_files(
        args.directory, args.limit
    )
    
    # Report large files
    if large_files:
        print("🚨 LARGE FILES FOUND (will cause GitHub issues):")
        print("-" * 60)
        
        # Sort by size
        large_files.sort(key=lambda x: x['size_mb'], reverse=True)
        
        for f in large_files:
            status = "❌ BLOCKED" if f['size_mb'] >= GITHUB_FILE_LIMIT_MB else "⚠️  WARNING"
            print(f"{status}: {f['path']}")
            print(f"         Size: {f['size_mb']:.1f} MB")
            print()
        
        # Count blocked vs warning
        blocked = sum(1 for f in large_files if f['size_mb'] >= GITHUB_FILE_LIMIT_MB)
        warnings = len(large_files) - blocked
        
        print("-" * 60)
        print(f"Total: {blocked} blocked (>100MB), {warnings} warnings (>{args.limit}MB)")
    else:
        print("✅ No files exceed the size threshold!")
    
    # Report by file type
    print("\n" + "=" * 60)
    print("SIZE BY FILE TYPE:")
    print("-" * 60)
    
    sorted_types = sorted(file_types.items(), key=lambda x: x[1]['size'], reverse=True)
    for ext, data in sorted_types[:15]:
        if data['size'] >= 1:
            print(f"  {ext:15} {data['count']:5} files  {data['size']:10.1f} MB")
    
    # Total size warning
    print("\n" + "=" * 60)
    total_gb = get_size_gb(total_size * 1024 * 1024)
    if total_gb > GITHUB_REPO_LIMIT_GB:
        print(f"⚠️  Total size: {total_gb:.2f} GB (exceeds recommended 1 GB)")
    else:
        print(f"✅ Total size: {total_size:.1f} MB ({total_gb:.2f} GB)")
    print("=" * 60)
    
    # Suggestions
    if large_files:
        print("\n💡 SUGGESTIONS:")
        print("-" * 60)
        
        # Check for common large file types
        large_extensions = set(f['extension'] for f in large_files)
        
        if '.pt' in large_extensions or '.pth' in large_extensions or '.bin' in large_extensions:
            print("• Model checkpoints (.pt, .pth, .bin):")
            print("  → Add to .gitignore")
            print("  → Use Git LFS or host separately (HuggingFace, Google Drive)")
        
        if '.json' in large_extensions:
            print("• Large JSON files:")
            print("  → Add to .gitignore if data files")
            print("  → Use Git LFS if must be tracked")
        
        if '.pkl' in large_extensions or '.pickle' in large_extensions:
            print("• Pickle files:")
            print("  → Add to .gitignore")
            print("  → Regenerate from code if needed")
        
        if '.db' in large_extensions or '.sqlite' in large_extensions:
            print("• Database files:")
            print("  → Add to .gitignore")
            print("  → Provide setup scripts to recreate")
        
        print("\n• General: Add this to your .gitignore:")
        print("  → See .gitignore in this repository for comprehensive list")

if __name__ == '__main__':
    main()