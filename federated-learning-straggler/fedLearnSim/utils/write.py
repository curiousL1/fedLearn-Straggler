def WriteToTxt(string, filename, filepath=""):
    file = open(filepath+filename+".txt", 'a')
    file.write(string)
    file.close()
