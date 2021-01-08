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