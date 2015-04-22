CXXFLAGS = -g -Wall -std=c99
CFLAGS = -g -Wall -lm -std=c99

all: heat_serial 

serial_objects = heat_serial.o creategrid.o stepper.o boundaryconditions.o
heat_serial: $(serial_objects)
	$(CC) -o $@ $^ $(CFLAGS)

clean:
	$(RM) *.o
	$(RM) .depend






