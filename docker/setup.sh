#!/bin/bash
# run with
# 'sudo su'
# 'source setup.sh'
# to force variables to be exported in the same shell
source ./root_setup.sh
su ubuntu << 'eof'
source ./user_setup.sh
eof

echo "cleaning setup files"
rm ./*setup.sh
rm ./*.md

echo "setup complete... rebooting now"
reboot now
