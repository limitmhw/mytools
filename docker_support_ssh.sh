apt install openssh -y
apt install openssh-server -y 


vim /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes

passwd root


systemctl restart ssh
/etc/init.d/ssh restart
