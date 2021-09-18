IDIRS = -I .
LIBS = -lGLFW
TARGET = artemis

artmes:
	g++ main.cpp glad.c $(IDIRS) $(LIBS) -o $(TARGET)
