import os

def get_last_folder_id(parent_dir):
    
    if not os.path.exists(parent_dir):
        print(f"The directory {parent_dir} does not exist.")
        return

    # Find the highest numbered directory
    highest_num = 0
    for item in os.listdir(parent_dir):
        if os.path.isdir(os.path.join(parent_dir, item)) and item.isdigit():
            highest_num = max(highest_num, int(item))
            
    return highest_num

def create_model_dir(parent_dir):
   
    highest_num = get_last_folder_id(parent_dir)
    last_dir_path = os.path.join(parent_dir, str(highest_num))
    model_exists = check_existing_model(last_dir_path)
    if model_exists:
        next_dir_num = highest_num + 1
        next_dir_path = os.path.join(parent_dir, str(next_dir_num))
        os.makedirs(next_dir_path, exist_ok=True)
        print(f"Created directory: {next_dir_path}")
        return next_dir_path
    else: 
        print(f"Using existing directory: {last_dir_path}")
        return last_dir_path

def check_existing_model(parent_dir):

    highest_num = get_last_folder_id(parent_dir)
            
    if highest_num == 0:
        return None
    
    last_dir_path = os.path.join(parent_dir, str(highest_num))

    for filename in os.listdir(last_dir_path):
        if filename.endswith(".pth") and os.path.isfile(os.path.join(last_dir_path, filename)):
            return os.path.join(last_dir_path, filename)
        
    return None

def create_seg_dir(parent_dir):
    highest_num = get_last_folder_id(parent_dir)
    if highest_num == 0:
        print(f"No directory found in {parent_dir}")
        return 
    seg_dir = os.path.join(parent_dir, str(highest_num), "seg")
    os.makedirs(seg_dir, exist_ok=True)
    print(f"Created directory: {seg_dir}")
    return seg_dir
