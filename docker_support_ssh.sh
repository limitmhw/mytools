apt install openssh -y
apt install openssh-server -y 


vim /etc/ssh/sshd_config
systemctl restart ssh
/etc/init.d/ssh restart
