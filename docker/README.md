# docker

## 1.安装
按照官方文档安装

https://docs.docker.com/engine/install/

## 2.常用命令
### 2.1 帮助命令
```
docker version # 版本
docker info # 显示docker的系统信息，包括镜像和容器的数量
docker 命令 --help # 查看帮助文档
```
### 2.2 镜像命令
```
docker images # 查看本地所有镜像
    [-a]  隐藏镜像也显示
    [-q]  只显示镜像id
    [-aq] 显示所有的镜像的id

docker search # 在仓库中搜索镜像

docker pull # 下载镜像

docker rmi 镜像id # 删除镜像
    [-f] 在创建了容器的情况下也强制删除镜像
    [-f $(docker images -aq)]  把所有的镜像的id作为参数传到-f后面  即删除本地所有镜像
```
### 2.3 容器命令
```
docker run 镜像id # 新建容器并启动

docker ps # 列出所有运行的容器 
    [-a] 显示所有容器 包括没有运行的
    [-q] 只显示容器id

docker rm 容器id # 删除指定容器

docker start 容器id # 启动容器

docker restart 容器id # 重启容器

docker stop 容器id # 停止当前正在运行的容器

docker kill 容器id # 强制停止当前容器

```
### 2.4 其他命令
