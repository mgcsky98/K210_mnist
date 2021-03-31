import sensor,lcd,image
import KPU as kpu
lcd.init()
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_windowing((224, 224))	#set to 224x224 input
sensor.set_hmirror(0)				#flip camera
task = kpu.load(0x300000)			#load model from flash address 0x200000
a=kpu.set_outputs(task, 0, 10,1,1)
sensor.run(1)
while True:
    img = sensor.snapshot()
    lcd.display(img,oft=(0,0))		#display large picture
    img1=img.to_grayscale(1)		#convert to gray
    img2=img1.resize(32,32)			#resize to mnist input 32x32
    a=img2.invert()					#invert picture as mnist need
    a=img2.strech_char(1)			#preprocessing pictures, eliminate dark corner
    lcd.display(img2,oft=(240,32))	#display small 32x32 picture
    a=img2.pix_to_ai();				#generate data for ai
    fmap=kpu.forward(task,img2)		#run neural network model
    plist=fmap[:]					#get result (10 digit's probability)
    pmax=max(plist)					#get max probability
    max_index=plist.index(pmax)		#get the digit
    lcd.draw_string(224,0,"%d: %.3f"%(max_index,pmax),lcd.WHITE,lcd.BLACK)

