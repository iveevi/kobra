IDIRS = -I .
LIBS = -lGLFW
TARGET = artmetis

artmes:
	g++ main.cpp glad.c $(IDIRS) $(LIBS) -o $(TARGET)
