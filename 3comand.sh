lvremove /dev/pve/data -y
lvresize -l +100%FREE /dev/pve/root
resize2fs /dev/mapper/pve-root
