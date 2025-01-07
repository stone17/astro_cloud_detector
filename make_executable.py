import PyInstaller.__main__
import os
import shutil
import subprocess

def build_exe(script_name, icon_path=None, additional_data=None, hidden_imports=None, excludes=None):
    """Builds a single executable using PyInstaller."""

    try:
        pyinstaller_args = [
            script_name,
            #'--onefile',      # Create a single executable
            #'--windowed',     # Suppress console window
            '--name={}'.format(os.path.splitext(script_name)[0]),
        ]

        if icon_path:
            pyinstaller_args.append('--icon={}'.format(icon_path))

        if additional_data:
            for data_item in additional_data:
                pyinstaller_args.extend(['--add-data', data_item])

        if hidden_imports:
            for hidden_import in hidden_imports:
                pyinstaller_args.extend(['--hidden-import', hidden_import])
        
        if excludes:
            for exclude in excludes:
                pyinstaller_args.extend(['--exclude-module', exclude])

        PyInstaller.__main__.run(pyinstaller_args)

        # Move the executable to the parent directory
        exe_name = os.path.splitext(script_name)[0] + ".exe"
        dist_path = os.path.join("dist", exe_name)
        output_path = os.path.join("..", exe_name)

        # Remove existing file if present before moving
        if os.path.exists(output_path):
            os.remove(output_path)
            
        shutil.move(dist_path, output_path)
        shutil.rmtree("build")
        shutil.rmtree("dist")
        shutil.rmtree("__pycache__") # remove the pycache
        print(f"Executable {exe_name} created successfully at {output_path}!")

    except subprocess.CalledProcessError as e:
        print(f"PyInstaller Error for {script_name}:")
        if e.stderr:
            print(e.stderr.decode())  # Print the error output from PyInstaller
        else:
            print("PyInstaller failed with no stderr output.")
    except Exception as e:
        print(f"General Error building executable for {script_name}: {e}")

if __name__ == "__main__":
    # Example usage for watcher_gui.py
    build_exe("cloud_watcher_gui.py",
              additional_data=[],
              hidden_imports=[],
              excludes=["pytest"])  # Add data files here if needed and exclude pytest

    # Example usage for training_gui.py
    build_exe("cloud_training_gui.py",
              additional_data=[], # Add data files here if needed
              hidden_imports=["sklearn"],
              excludes=["pytest"]) # Corrected hidden imports and added sklearn and exclude pytest