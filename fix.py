with open("scripts/3d_reconstruction/reconstruct_3d_b1k.py", "r") as f:
    content = f.read()

# find print("\nRunning VGGT reconstruction...")
save_images_code = """
    print(f"\\nSaving images to {output_dir}/images...")
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    for i, img in enumerate(all_images_list):
        img.save(os.path.join(images_dir, f"{i:04d}.png"))
"""

if "Saving images to" not in content:
    content = content.replace('    print("\\nRunning VGGT reconstruction...")', save_images_code + '\n    print("\\nRunning VGGT reconstruction...")')

with open("scripts/3d_reconstruction/reconstruct_3d_b1k.py", "w") as f:
    f.write(content)
