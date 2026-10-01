# Run the baby-cry test page on a server

This page listens on your PC at `127.0.0.1` unless you set a host. On a VPS, use a new folder and listen on all network interfaces.

The sound clips are large, so they are not stored in git. Copy `data\ESC-50-master` and `output\model` onto the server next to the code.

## 1. Create a folder

On the Windows server, create:

`C:\babycrydetection`

Put this project inside that folder so `app_ui\app.py` is at `C:\babycrydetection\app_ui\app.py`.

## 2. Install and start

Open PowerShell in `C:\babycrydetection`.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:BABY_CRY_HOST = "0.0.0.0"
$env:BABY_CRY_PORT = "5000"
$env:BABY_CRY_NO_BROWSER = "1"
.\.venv\Scripts\python.exe app_ui\app.py
```

Leave that window open. The page is then at:

`http://YOUR_SERVER_IP:5000`

Port 5000 must be allowed through the server firewall.

## 3. Sign in to the server

Use an SSH client and sign in as the administrator account for the VPS. Do not put the password in this repository. Change the password if it has been shared in a chat or email.
