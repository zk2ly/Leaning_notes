# git
- [git](#git)
  - [1.ubuntu安装git](#1ubuntu安装git)
  - [2.配置git](#2配置git)
  - [3.git操作](#3git操作)
  - [4.其他](#4其他)
## 1.ubuntu安装git
```
sudo apt-get update # 访问软件源  更新软件列表
sudo apt-get upgrade # 更新软件

# 软件源 可以配置国内的比较快
cat /etc/apt/sources.list

apt install git # 安装git
git --version # 查看版本
```
## 2.配置git
配置全局的github账号信息
```
git config --global user.name "username"
git config --global user.email "email"

git config -l # 查看配置信息 
```
配置ssh
```
ssh-keygen -t rsa -C "email" # rsa算法生成密钥 一直回车
```

/root/.ssh/下生成私钥id_rsa和公钥id_rsa.pub
```
cat id_rsa.pub  # 复制公钥添加到github的SSH Keys
ssh -T git@github.com # 检测是否可以通过ssh连接到github 
```

## 3.git操作
```
git init # 初始化一个空仓库

git clone https:\\ # https协议克隆一个已经有的仓库 推送时要输入账号密码
git remote -v # 查看当前https协议连接的仓库

git clone git@github.com: # SSH协议克隆一个已有的仓库 不需要输入账号密码
```

```
git add file  # 添加文件到暂存区
git add . # 添加工作区所有文件到暂存区

git commit -m "xxx" # 暂存区代码提交到仓库

git push origin main # 推送到远程仓库

git status # 显示有变更的文件
```

## 4.其他
需要用到时，查阅并补充

https://gitee.com/all-about-git

