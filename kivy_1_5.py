from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
#from kivy.uix.floatlayout import FloatLayout
from kivy.uix.button import Button
from kivy.graphics import Line
from kivy.uix.label import Label
#from kivy import *
import numpy as np



class WidgetExample(GridLayout):
	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		#x=np.random.randint(0,9)
		d= DrawInput()
		j = str(d.x)
		l=Label(text=j, size_hint=(.2,1))
		self.add_widget(l)
		self.add_widget(d)
	

	
class DrawInput(Widget):
	x=np.random.randint(0,9)
	r = np.random.randint(10000, 999999)
	def on_touch_down(self, touch):
		with self.canvas:
			touch.ud["line"]= Line(points=(touch.x, touch.y), width=10)
		
	def on_touch_move(self, touch):
		touch.ud["line"].points += (touch.x, touch.y)
		
	def on_touch_up(self, touch):
		self.export_to_png(f'{self.x}/image{self.r}.png')


		
class SimplyKivy4(App):
	def build(self):
		layout = WidgetExample()
		return layout
		
if __name__ == "__main__":
	
	SimplyKivy4().run()
	