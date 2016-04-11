
import MySQLdb as mysql
import csv

def mysql_test():

	print "hello"

	db = mysql.connect(host='localhost',user='root',passwd='4727238',db='testdb')

	# prepare a cursor object using cursor() method
	cursor = db.cursor()
	sql = "DROP TABLE IF EXISTS IMAGE"
	print "hello"
	print cursor.execute(sql)

	# Create table as per requirement
	sql = "CREATE TABLE IMAGE (PICNAME CHAR(20))"

	cursor.execute(sql)

	# Prepare SQL query to INSERT a record into the database.
	sql = """INSERT INTO IMAGE(PICNAME)
			 VALUES (1.0)"""

	try:
		# Execute the SQL command
		cursor.execute(sql)
		# Commit your changes in the database
		db.commit()
	except:
		# Rollback in case there is any error
		db.rollback()

	# Fetch a single row using fetchone() method.
	cursor.execute("SELECT * FROM grayscales")
	data = cursor.fetchone()

	print "data = " + str(data)

	# disconnect from server
	db.close()


def csv_test():
	myfile = open("./testcsv.csv",'wb')
	wr = csv.writer(myfile, quoting=csv.QUOTE_NONE)
	alist = [1,2,3,4,5,6,7,8,43,12234,23,4,6,8,4,4,3,234,23,56,6,5,63,57,37,556,23,667,455,675,67,45567,45,67,4567]
	wr.writerow(alist)
	myfile.close()

	myfile = open("./testcsv.csv",'rt')
	rr = csv.reader(myfile)
	newlist = rr[0]
	print newlist
	myfile.close()


if __name__ == "__main__":
	csv_test()