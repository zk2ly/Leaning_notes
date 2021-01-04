# git

## 1.ubuntu安装git
```
sudo apt-get update # 访问软件源  更新软件列表
sudo apt-get upgrade # 更新软件

# 软件源
cat /etc/apt/sources.list

apt install git # 安装git
git --version # 查看版本
```
```
## Note, this file is written by cloud-init on first boot of an instance
## modifications made here will not survive a re-bundle.
## if you wish to make changes you can:
## a.) add 'apt_preserve_sources_list: true' to /etc/cloud/cloud.cfg
##     or do the same in user-data
## b.) add sources in /etc/apt/sources.list.d
## c.) make changes to template file /etc/cloud/templates/sources.list.tmpl

# See http://help.ubuntu.com/community/UpgradeNotes for how to upgrade to
# newer versions of the distribution.
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic main
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic main

## Major bug fix updates produced after the final release of the
## distribution.
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates main
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates main

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu
## team. Also, please note that software in universe WILL NOT receive any
## review or updates from the Ubuntu security team.
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic universe
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic universe
deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates universe
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates universe

## N.B. software from this repository is ENTIRELY UNSUPPORTED by the Ubuntu 
## team, and may not be under a free licence. Please satisfy yourself as to
## your rights to use the software. Also, please note that software in 
## multiverse WILL NOT receive any review or updates from the Ubuntu
## security team.
# deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic multiverse
# deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic multiverse
# deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates multiverse
# deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-updates multiverse

## Uncomment the following two lines to add software from the 'backports'
## repository.
## N.B. software from this repository may not have been tested as
## extensively as that contained in the main release, although it includes
## newer versions of some applications which may provide useful features.
## Also, please note that software in backports WILL NOT receive any review
## or updates from the Ubuntu security team.
# deb http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src http://mirrors.cloud.aliyuncs.com/ubuntu/ bionic-backports main restricted universe multiverse

## Uncomment the following two lines to add software from Canonical's
## 'partner' repository.
## This software is not part of Ubuntu, but is offered by Canonical and the
## respective vendors as a service to Ubuntu users.
# deb http://archive.canonical.com/ubuntu bionic partner
# deb-src http://archive.canonical.com/ubuntu bionic partner

deb http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security main
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security main
deb http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security universe
deb-src http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security universe
# deb http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security multiverse
# deb-src http://mirrors.cloud.aliyuncs.com/ubuntu bionic-security multiverse
```

## 2.配置git
配置全局的github账号信息
```
git config --global user.name "username"
git config --global user.email "email"

git config -l # 查看配置信息

ssh-keygen -t rsa -C "email" # 生成密钥 一直回车
```
/root/.ssh/下生成私钥id_rsa和公钥id_rsa.pub
```
cat id_rsa.pub  # 复制公钥添加到github的SSH Keys
ssh -T git@github.com # 配置私钥和公钥
```

## 3.git操作
```
git init # 初始化一个空仓库
git clone www.xxx.com # 克隆一个已经有的仓库

git add file  # 添加文件到暂存区

git commit -m "xxx" # 暂存区代码提交到仓库

git push origin main # 推送到远程仓库

```

