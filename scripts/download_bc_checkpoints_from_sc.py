"""
Select and download checkpoints from supercloud
"""

import os
import paramiko
import numpy as np
import datetime
import stat

def download_dir_via_sftp(sftp, remote_dir, local_dir):
    os.makedirs(local_dir, exist_ok=True)
    for item in sftp.listdir(remote_dir):
        remote_item = os.path.join(remote_dir, item)
        local_item = os.path.join(local_dir, item)

        fileattr = sftp.lstat(remote_item)
        if stat.S_ISDIR(fileattr.st_mode):
            download_dir_via_sftp(sftp, remote_item, local_item)
        else:
            sftp.get(remote_item, local_item)

user_id = input("Your supercloud user id (1) azhao1, (2) yma, (2) lwang: ")
if user_id == "1":
    user = "azhao1"
elif user_id == "2":
    user = "yma"
elif user_id == "3":
    user = "lwang"
else:
    print("Invalid user id")
    exit()

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('txe1-login.mit.edu', username=user)

sftp = client.open_sftp()
print("Connected to supercloud")

remote_p = 'src/FoundationTactile/TheProbe/ProbingPanda/checkpoints'
cps = []
cps_mtime = [] # modified time
for checkpoint in sftp.listdir(remote_p):
    cp_path = os.path.join(remote_p, checkpoint)
    try:
        cps_mtime.append(sftp.stat(os.path.join(cp_path, "policy_model.pt")).st_mtime)
    except:
        continue
    cps.append(cp_path)

# sort by modified time
cps_mtime = np.array(cps_mtime)
cps = np.array(cps)
cps = cps[np.argsort(cps_mtime)][::-1]
cps_mtime = np.sort(cps_mtime)[::-1]

# print a list
print("Checkpoints in supercloud listed below")
for i, (cp, cp_mtime) in enumerate(zip(cps, cps_mtime)):
    print(f"id:{i+1}\t{os.path.basename(cp)}\t{datetime.datetime.fromtimestamp(cp_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

os.makedirs("checkpoints_supercloud", exist_ok=True)
print("="*80)
print("Type in the id of the checkpoint you want to download. '' to download the latest, 'q' to quit, 'i-j' to download a range of checkpoints, 'all' to download all")
print("For example, '', '2', '1-3', 'q', 'all'")
print("="*80)
inp = input("Your input: ")
if inp == "q":
    print("Quit")
    exit()
elif inp == "all":
    for cp in cps:
        local_p = os.path.join("checkpoints_supercloud", os.path.basename(cp))
        print(f"Downloading {cp} to {local_p}")
        download_dir_via_sftp(sftp, cp, local_p)
    print("All checkpoints downloaded")
    exit()
elif "-" in inp:
    i, j = map(int, inp.split("-"))
    for cp in cps[i-1:j]:
        local_p = os.path.join("checkpoints_supercloud", os.path.basename(cp))
        print(f"Downloading {cp} to {local_p}")
        download_dir_via_sftp(sftp, cp, local_p)
    print(f"Checkpoints {i} to {j} downloaded")
    exit()

if inp == "":
    cp = cps[0]
else:
    cp = cps[int(inp)-1]
local_p = os.path.join("checkpoints_supercloud", os.path.basename(cp))
print(f"Downloading {cp} to {local_p}")
download_dir_via_sftp(sftp, cp, local_p)
print("Downloaded")
