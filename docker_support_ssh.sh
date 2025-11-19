apt install openssh -y
apt install openssh-server -y 

systemctl restart ssh
/etc/init.d/ssh restart
