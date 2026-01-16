apt install openssh -y
apt install openssh-server -y 


vim /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes
Port 2222


passwd root


systemctl restart ssh
/etc/init.d/ssh restart











useradd -m -s /bin/bash l
passwd l

usermod -aG sudo l


grep -R "AllowUsers\|DenyUsers" /etc/ssh/
如果有：
AllowUsers xxx
👉 把 l 加进去
