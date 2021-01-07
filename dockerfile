FROM centos
MAINTAINER zhangk<377351842@qq.com>

ENV MYPATH /usr/local
WORKDIR $MYPATH

RUN yum -y install vim

EXPOSE 80

CMD /bin/bash
