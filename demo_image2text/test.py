# %%
from bs4 import BeautifulSoup as Soup

picture_html = """<div id= "left-bar" >""" + """</div> <div id= "right-bar" >""" + "</div>"
with open('templates/home.html', 'rb') as f:
        home = Soup(f.read(), 'html.parser')

# cls_h2 = home.find_all("h2", string=" Image Classification ")[0]
begin_tag = home.find("div", {"style":"padding-left:16px"})
div1 = home.new_tag('div')
print(type(div1))
div1['id'] = "left-bar"
div2 = home.new_tag('div')
div2['id'] = "right-bar"
div2.string = '10'

begin_tag.append(div1)
begin_tag.append(div2)
# cls_h2.append(div1)
# cls_h2.append(div2)

print(str(home))
# print(type(home))

# %%
from bs4 import BeautifulSoup

# Input string
input_string = '<div id= "left-bar" ><p> house_finch </p> <picture> <img src= "/static/STanager-Shapiro.jpg"  height="300" width="400"> </picture></div> <div id= "right-bar" ><li> house_finch : 1 </li></div>'

# Create a BeautifulSoup object from the input string
soup = BeautifulSoup(input_string, 'html.parser')

# Print the BeautifulSoup object
print(soup.prettify())
# %%

with open('templates/home.html', 'rb') as f:
    home = Soup(f.read(), 'html.parser')

begin_tag = home.find("div", {"style":"padding-left:16px"})
begin_tag.append(soup)

home
# %%
