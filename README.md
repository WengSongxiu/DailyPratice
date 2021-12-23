# 项目名称

[TOC]

## Structure

| 文件夹名         | 说明                                          |
| :--------------- | :-------------------------------------------- |
| bin              | 存放项目的一些可执行二进制文件，cmd/sh等      |
| conf             | 存放配置文件                                  |
| core             | 存放项目的所有源代码(核心代码）和单元测试代码 |
| data             | 存放输入输出数据                              |
| lib              | 库文件，存放其他依赖模块和包                  |
| log              | 存放日志文件                                  |
| model            | 存放模型文件                                  |
| docs             | 存放文档                                      |
| README.md        | 工程说明文档                                  |
| Jenkinsfile      | 构建文件                                      |
| requirements.txt | 依赖清单                                      |

## Quick Start

1.工程导进去之后，别人怎么跑起来（说明需要别人怎么做，具体流程是什么）

## Requirements

执行源代码需要按照以下软件。

* Python 3.x
* requirements.txt

```shell
# 安装依赖管理工具
pip install pipreqs
# 导出工程依赖[工程目录]
pipreqs ./ --encoding utf-8
# 安装依赖
pip install -r requirements.txt
```

## Train

1.怎么训练

## Predict

1.怎么预测

## Docker

容器化操作说明

###  1、 编写Dockerfile

Dockerfile的模板，Dockerfile、requirement.txt和Jenkinsfile 文件统一放到项目根目录。

```shell
FROM python:3.7
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/usr/src/app:$PYTHONPATH
CMD ["python","./core/framework/interface.py"]
```

### 3、 制作镜像

```shell
docker build -t {镜像名称}:{镜像tag} .
```

### 4、 导出镜像【可选】

```shell
docker save -o {镜像名称}.tar {镜像名称}:{镜像tag}
```

### 5、 导入镜像【可选】

```shell
docker load -i {镜像名称}.tar
```

### 6、 启动镜像

```shell
docker run -d -p 宿主机端口:容器端口 --rm --name {容器名称} {镜像名称}:{镜像tag}
```

### 7、 进入容器

```shell
docker exec -it {容器id} /bin/bash
```

## Interface（API）

说明接口化相关

## Deployment

1.说明部署相关
