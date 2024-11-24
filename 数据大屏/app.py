#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Describe: 使用 Flask 构建一个简单的 Web 应用，展示不同类型的数据

from flask import Flask, render_template
from data import *  # 导入数据模块，假设其中定义了 SourceData, CorpData, JobData 类

# 创建 Flask 应用实例
app = Flask(__name__)

@app.route('/')
def corp():
    """
    企业页面路由，展示 CorpData 数据
    """
    data = CorpData()  # 实例化 CorpData 类，获取数据
    return render_template('index.html', form=data, title=data.title)  # 渲染模板并传递数据


if __name__ == "__main__":
    # 启动 Flask 应用
    app.run(host='127.0.0.1', debug=True)  # 在本地主机上运行应用，并开启调试模式