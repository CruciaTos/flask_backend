import os
import subprocess

# Step 1: Generate requirements.txt
print("Generating requirements.txt...")
subprocess.run("pip freeze > requirements.txt", shell=True)

# Step 2: Create Procfile
procfile_content = "web: python app.py"

print("Creating Procfile...")
with open("Procfile", "w") as procfile:
    procfile.write(procfile_content)

print("✅ requirements.txt and Procfile created successfully!")

# Step 3: Git Commit and Push
print("Adding files to Git...")
subprocess.run("git add requirements.txt Procfile", shell=True)
subprocess.run('git commit -m "Added requirements.txt and Procfile"', shell=True)
subprocess.run("git push origin main", shell=True)

print("✅ Changes pushed to GitHub! Try deploying again on Railway.")
