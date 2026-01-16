apt install openssh -y
apt install openssh-server -y 


vim /etc/ssh/sshd_config
PermitRootLogin yes
PasswordAuthentication yes


systemctl restart ssh
/etc/init.d/ssh restart
