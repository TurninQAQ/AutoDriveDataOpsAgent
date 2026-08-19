import yaml
import os
import argparse


def load_yaml_with_header(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    if content.startswith('%YAML'):
        lines = content.split('\n')
        content = '\n'.join(lines[2:])
    return yaml.safe_load(content)


def add_td_to_yaml(yaml_path, td_value):
    data = load_yaml_with_header(yaml_path)
    
    data['td'] = td_value
    
    with open(yaml_path, 'w') as f:
        f.write('%YAML:1.0\n')
        f.write('---\n')
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"Updated: {yaml_path} -> td: {td_value}")


def main():
    parser = argparse.ArgumentParser(description='Add td field to camera intrinsic YAML files')
    parser.add_argument('--td', type=float, default=0.0,
                        help='Time delay value to set (default: 0.9)')
    parser.add_argument('--dir', type=str,
                        default='data/calib/camera/intrinsic',
                        help='Directory containing intrinsic YAML files')
    
    args = parser.parse_args()
    
    td_value = args.td
    intrinsic_dir = args.dir
    
    if not os.path.isdir(intrinsic_dir):
        print(f"Error: Directory not found: {intrinsic_dir}")
        exit(1)
    
    yaml_files = [f for f in os.listdir(intrinsic_dir) if f.endswith('.yaml')]
    
    if not yaml_files:
        print(f"No YAML files found in: {intrinsic_dir}")
        exit(0)
    
    print(f"Found {len(yaml_files)} YAML file(s):")
    for f in yaml_files:
        print(f"  - {f}")
    
    print(f"\nSetting td: {td_value} to all files...")
    
    for yaml_file in yaml_files:
        yaml_path = os.path.join(intrinsic_dir, yaml_file)
        add_td_to_yaml(yaml_path, td_value)
    
    print("\nDone!")

# sample2
# python tools/tools_calib/add_td_to_intrinsic.py --td 0.9 --dir /home/cidi/data_pipeline/data_set_SQ_1/2026-07-14-xw/record_CLOUD_MAPPING_2026-07-14_170929-csc/calibration/camera/intrinsic
# python tools/tools_calib/add_td_to_intrinsic.py --td 0.9 --dir /home/cidi/data_pipeline/data_set_SQ_1/2026-07-14/record_CLOUD_MAPPING_2026-07-14_091708/calibration/camera/intrinsic
# python tools/tools_calib/add_td_to_intrinsic.py --td 0.9 --dir /home/cidi/data_pipeline/data_set_SQ_1/2026-07-14-xw/record_CLOUD_MAPPING_2026-07-14_171415/calibration/camera/intrinsic
if __name__ == "__main__":
    main()