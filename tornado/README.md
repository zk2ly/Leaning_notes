# Tornado

学习书籍：

http://demo.pythoner.com/itt2zh/

简单例子1
```python
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
from tornado.options import define, options

# 定义默认参数  默认端口是8000 可以在运行文件时用 --port=8001来更改 必须是int型 也可以用--help查看help中写的内容
define("port", default=8000, help="run on the given port", type=int)

# 请求处理类 这里定义了get方法 将对http的get请求做出处理
class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        greeting = self.get_argument('greeting', 'Hello')  # 查询请求中greeting的值 没有默认为Hello
        self.write(greeting + ', friendly user!')  # 把一个字符串写入http相应中

if __name__ == "__main__":
    tornado.options.parse_command_line()  # 解析命令行
    app = tornado.web.Application(handlers=[(r"/", IndexHandler)])  # 定义handlers参数构建app实例，handlers是一个元组组成的列表,正则表达式匹配的内容会传入app实例
    http_server = tornado.httpserver.HTTPServer(app)  # app实例传递给HTTPServer
    http_server.listen(options.port)  # 在指定端口监听
    tornado.ioloop.IOLoop.instance().start()  # 创建一个循环实例 并运行

"""
curl http://localhost:8000/
Hello, friendly user!

curl http://localhost:8000/?greeting=Salutations    get请求传递参数用? 参数间用&连接 受url限制不能传输太多数据 而且必须ascii字符 不能编码转换中文
Salutations, friendly user!
"""
```
简单例子2
```python

import textwrap

import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)

class ReverseHandler(tornado.web.RequestHandler):
    def get(self, input):
        self.write(input[::-1]+'\n')

class WrapHandler(tornado.web.RequestHandler):
    def post(self):
        text = self.get_argument('text')
        width = self.get_argument('width', 40)
        self.write(textwrap.fill(text, int(width))+'\n')
        
if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(
        handlers=[
            (r"/reverse/(\w+)", ReverseHandler),  # 匹配保存括号中的多个字符 传递给实例
            (r"/wrap", WrapHandler)  
        ]
    )
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()

"""
curl http://localhost:8000/reverse/8000
0008

curl http://localhost:8000/wrap -d text=Lorem+ipsum+dolor+sit+amet,+consectetuer+adipiscing+elit.  # post请求传递参数用-d 
Lorem ipsum dolor sit amet, consectetuer
adipiscing elit.
"""
```
