from flask import Flask, render_template, request, redirect
import pandas as pd
import plotly.graph_objs as go
import plotly.offline as pyo
#import mysql.connector as msql
#from mysql.connector import Error
#import psycopg2
tvlists = Flask(__name__)

# def connection():
#     s = 'localhost' # server(host) name 
#     d = 'BestBuy' 
#     u = 'postgres' # login user
#     p = '123456' # login password
#     conn = psycopg2.connect(host=s, user=u, password=p, database=d)
#     return conn


## This code sets up two route handlers: one for the main page ("/") 
# and another for retrieving and displaying TV listings ("/tvlist"). 
# The second route handler handles both GET and POST requests and interacts with a database
# To fetch TV listings based on filter criteria or display all listings.
# This code executes an SQL query to retrieve TV listings
# from a database table based on the specified page number. 


@tvlists.route("/")
def main_1():
    return render_template("index.html")

# @tvlists.route("/tvlist", methods = ['GET', 'POST'])
# def get_tvlist():
#     tvs = [ ]
#     search = [ ]
#     count = 0
#     conn = connection()
#         # if conn.is_connected():

#     if request.method == 'POST':
#         brand = request.form["brand"]
#         type = request.form["type"]
#         size = request.form["size"]
#         pixel = request.form["pixel"]
#         cursor = conn.cursor()
#         cursor_count = conn.cursor()
#         cursor_count.execute("SELECT count(*) FROM tv_bestbuy Where brand ~ '" + brand + "' AND tv_type ~ '" + type + "' AND tv_size ~ '" + size + "' AND tv_pixels ~ '" + pixel + "' ")
#         for row in cursor_count.fetchall():
#             count = row[0]
#         cursor.execute("SELECT * FROM tv_bestbuy Where brand ~ '" + brand + "' AND tv_type ~ '" + type + "' AND tv_size ~ '" + size + "' AND tv_pixels ~ '" + pixel + "' ORDER BY index limit 10 offset 0 ")
#         for row in cursor.fetchall():
#             tvs.append({"id": row[0], "description": row[1], "reviews": row[2], "price": row[3], "discount": row[4], "brand": row[5], "tv_type": row[6], "tv_size": row[7], "tv_pixel": row[8]})
#         conn.close()
#         return render_template("tvlist.html", tvs = tvs, count = count, page = 1)
#     cursor_count = conn.cursor()
#     cursor_count.execute("SELECT count(*) FROM tv_bestbuy ")   
#     for row in cursor_count.fetchall():
#         count = row[0]
#     cursor = conn.cursor()

#     # cursor.execute("SELECT * FROM tv_bestbuy ORDER BY id limit " + str(int(page) * 10) + ", 10 ")
#     cursor.execute("SELECT * FROM tv_bestbuy ORDER BY index limit 10 offset 0 ")
#     for row in cursor.fetchall():
#         tvs.append({"id": row[0], "description": row[1], "reviews": row[2], "price": row[3], "discount": row[4], "brand": row[5], "tv_type": row[6], "tv_size": row[7], "tv_pixel": row[8]})
#     conn.close()
#     return render_template("tvlist.html", tvs = tvs, count = count, page = 1)






# @tvlists.route("/get/<int:page>", methods = ['GET', 'POST'])
# def get_data(page):
#     # try:
#     tvs = [ ]
#     search = [ ]
#     count = 0
#     conn = connection()


#         # if conn.is_connected():
#     if request.method == 'POST':
#         brand = request.form["brand"]
#         type = request.form["type"]
#         size = request.form["size"]
#         pixel = request.form["pixel"]
#         cursor = conn.cursor()
#         cursor_count = conn.cursor()
#         cursor_count.execute("SELECT count(*) FROM tv_bestbuy Where brand ~ '" + brand + "' AND tv_type ~ '" + type + "' AND tv_size ~ '" + size + "' AND tv_pixels ~ '" + pixel + "' ")
#         for row in cursor_count.fetchall():
#             count = row[0]
#         cursor.execute("SELECT * FROM tv_bestbuy Where brand ~ '" + brand + "' AND tv_type ~ '" + type + "' AND tv_size ~ '" + size + "' AND tv_pixels ~ '" + pixel + "' ORDER BY index limit 10 offset 0 ")
#         for row in cursor.fetchall():
#             tvs.append({"id": row[0], "description": row[1], "reviews": row[2], "price": row[3], "discount": row[4], "brand": row[5], "tv_type": row[6], "tv_size": row[7], "tv_pixel": row[8]})
#         conn.close()
#         return render_template("tvlist.html", tvs = tvs, count = count, page = 1)
#     cursor_count = conn.cursor()
#     cursor_count.execute("SELECT count(*) FROM tv_bestbuy ")   
#     for row in cursor_count.fetchall():
#         count = row[0]
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM tv_bestbuy ORDER BY index limit 10 offset " + str(int(page-1) * 10) + " ")

#     # cursor.execute("SELECT * FROM tv_bestbuy ORDER BY id limit 0, 10 ")
#     for row in cursor.fetchall():
#         tvs.append({"id": row[0], "description": row[1], "reviews": row[2], "price": row[3], "discount": row[4], "brand": row[5], "tv_type": row[6], "tv_size": row[7], "tv_pixel": row[8]})
#     conn.close()
#     return render_template("tvlist.html", tvs = tvs, count = count, page = page)

 


# # GRAPH 1

# @tvlists.route("/graph1")
# def graph1():
#     # try:
#     tvs = []
#     conn = connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM tv_bestbuy")
#     tvs = cursor.fetchall()
#     # this is about plotly graph
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=[row[0] for row in tvs], y=[row[1] for row in tvs], mode='lines', name='description'))
#     # end 
#     conn.close()
#     # Display the figure in a web browser - it is
#     return render_template("graph1.html", rows = tvs)
 

#     # GRAPH 2

# @tvlists.route("/graph2")
# def graph2():
#     # try:
#     tvs = []
#     conn = connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM tv_bestbuy")
#     tvs = cursor.fetchall()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=[row[0] for row in tvs], y=[row[1] for row in tvs], mode='markers', name='description'))
#     conn.close()
#     # Display the figure in a web browser
#     # pyo.plot(fig, filename='graph1.html')
#     return render_template("graph2.html", rows = tvs)
   

#     # GRAPH 3

# @tvlists.route("/graph3")
# def graph3():
#     # try:
#     tvs = []
#     conn = connection()
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM tv_bestbuy")
#     tvs = cursor.fetchall()
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=[row[0] for row in tvs], y=[row[1] for row in tvs], mode='markers', name='description'))
#     conn.close()
#     # Display the figure in a web browser
#     # pyo.plot(fig, filename='graph1.html')
#     return render_template("graph3.html", rows = tvs)
    

@tvlists.route("/map")
def show_map():
    return render_template("map.html")

if(__name__ == "__main__"):
    tvlists.run()