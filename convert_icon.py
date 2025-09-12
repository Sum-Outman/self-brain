#!/usr/bin/env python3
"""
Icon Conversion Script
Converts 'Self Brain.bmp' to multiple formats for cross-platform AGI system.
"""

import os
import sys
import subprocess
import shutil

def check_and_install_pillow():
    """Check if Pillow is installed, install if not."""
    try:
        from PIL import Image
        print("✓ Pillow is already installed")
        return True
    except ImportError:
        print("Pillow not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pillow"])
            from PIL import Image
            print("✓ Pillow installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install Pillow. Please install it manually:")
            print("  pip install pillow")
            return False

def convert_icon():
    """Convert the BMP icon to multiple formats."""
    try:
        from PIL import Image
        
        # Input file
        input_file = "Self Brain.bmp"
        
        if not os.path.exists(input_file):
            print(f"✗ Input file '{input_file}' not found")
            return False
        
        print(f"Processing: {input_file}")
        
        # Open the image
        with Image.open(input_file) as img:
            # Get original size
            width, height = img.size
            print(f"Original size: {width}x{height}")
            
            # Create icons directory
            os.makedirs("icons", exist_ok=True)
            
            # Target sizes for different platforms
            sizes = [16, 32, 48, 64, 128, 256]
            
            # Convert to PNG with multiple sizes
            print("\nConverting to PNG formats:")
            for size in sizes:
                # Resize maintaining aspect ratio
                resized_img = img.resize((size, size), Image.LANCZOS)
                output_file = f"icons/self_brain_{size}.png"
                resized_img.save(output_file, "PNG")
                print(f"✓ Created {output_file}")
            
            # Convert to ICO (Windows icon) - requires special handling
            print("\nConverting to ICO format:")
            ico_sizes = [(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
            ico_images = []
            
            for size in ico_sizes:
                resized_img = img.resize(size, Image.LANCZOS)
                ico_images.append(resized_img)
            
            ico_output = "icons/self_brain.ico"
            ico_images[0].save(ico_output, format='ICO', sizes=ico_sizes)
            print(f"✓ Created {ico_output}")
            
            # Convert to SVG (vector approximation - note: this creates a PNG embedded in SVG)
            # For true vector conversion, you'd need the original vector source
            print("\nCreating SVG approximation:")
            svg_output = "icons/self_brain.svg"
            with open(svg_output, 'w') as f:
                f.write(f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
  <image href="{input_file}" width="{width}" height="{height}"/>
</svg>''')
            print(f"✓ Created {svg_output} (approximation)")
            
            print(f"\n🎉 All conversions completed! Files saved in 'icons/' directory")
            return True
            
    except Exception as e:
        print(f"✗ Error during conversion: {e}")
        return False

def main():
    """Main function."""
    print("=" * 50)
    print("AGI System Icon Conversion Tool")
    print("=" * 50)
    
    if not check_and_install_pillow():
        return
    
    if convert_icon():
        print("\nRecommended file usage:")
        print("- self_brain.ico: Windows application icon")
        print("- self_brain_16.png: Browser favicon")
        print("- self_brain_32.png: Taskbar icon")
        print("- self_brain_64.png: Desktop shortcut")
        print("- self_brain_128.png: Application list")
        print("- self_brain_256.png: High-resolution displays")
        print("- self_brain.svg: Vector version for scaling")
    else:
        print("\nConversion failed. Please check the error messages.")

if __name__ == "__main__":
    main()