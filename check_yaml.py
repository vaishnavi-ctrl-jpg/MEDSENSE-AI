import yaml
import sys

def validate_medsense_yaml():
    try:
        with open("openenv.yaml", "r") as f:
            data = yaml.safe_load(f)
        
        # Critical checks for Nidz's task
        required_keys = ["name", "interface", "action_space", "docker"]
        for key in required_keys:
            if key not in data:
                print(f"❌ Missing Key: {key}")
                return
        
        if data['interface']['type'] != 'api':
            print("❌ Error: Interface type must be 'api'")
            return
            
        print("✅ YAML Logic is Correct!")
        print(f"🚀 API Base URL: {data['interface']['base_url']}")
        print(f"🐳 Docker Image: {data['docker']['image']}")

    except Exception as e:
        print(f"❌ YAML Syntax Error: {e}")

if __name__ == "__main__":
    validate_medsense_yaml()