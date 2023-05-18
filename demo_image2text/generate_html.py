import glob
from bs4 import BeautifulSoup as Soup

html_string_start = """
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>

<style>
#left-bar {
  position: fixed;
  display: table-cell;
  top: 100;
  bottom: 10;
  left: 10;
  width: 35%;
  overflow-y: auto;
}

#right-bar {
  position: fixed;
  display: table-cell;
  top: 100;
  bottom: 10;
  right: 10;
  width: 45%;
  overflow-y: auto;
}

</style>
<body>

<center><h1> Image Classification </h1></center>

<div id= "left-bar" >

"""


html_string_end = """

</body>
</html>

"""

def get_picture_html(path, tag):
    image_html = """<p> {tag_name} </p> <picture> <img src= "/static/{path_name}"  height="300" width="400"> </picture>"""
    return image_html.format(tag_name=tag, path_name=path)

def get_count_html(category, count):
    count_html = """<li> {category_name} : {count_} </li>"""
    return count_html.format(category_name = category, count_ = count)


def get_value_count(image_class_dict):
    count_dic = {}
    for category in image_class_dict.values():
        if category in count_dic.keys():
            count_dic[category] = count_dic[category]+1
        else:
            count_dic[category] = 1
    return count_dic


def generate_html(image_class_dict):
    picture_html = ""
    count_html = ""
    
    for image in image_class_dict.keys():
        picture_html += get_picture_html(path=image, tag= image_class_dict[image])
        
    value_counts = get_value_count(image_class_dict)
    
    for value in value_counts.keys():
        count_html += get_count_html(value, value_counts[value])
        
    # file_content = html_string_start + picture_html + """</div> <div id= "right-bar" >""" + count_html+ "</div>" +html_string_end
    picture_html = """<div id= "left-bar" >""" + picture_html + """</div> <div id= "right-bar" >""" + count_html+ "</div>"

    # Create a BeautifulSoup object from the input string
    picture_soup = Soup(picture_html, 'html.parser')
    
    with open('templates/home.html', 'rb') as f:
        home = Soup(f.read(), 'html.parser')
    
    begin_tag = home.find("div", {"style":"padding-left:16px"})
    begin_tag.append(picture_soup)

    #
    with open('templates/image_class.html', 'w') as f:
        f.write(str(home))


def generate_html_2(input_text, image_path):
    with open('templates/home.html', 'rb') as f:
        home = Soup(f.read(), 'html.parser')
    
    picture_html = f"""<picture> <img src= "{image_path[1:]}"  height="300" width="400"> </picture>"""
    picture_soup = Soup(picture_html, 'html.parser')

    text_html = "<p>" + input_text + "</p>"
    text_soup = Soup(text_html, 'html.parser')

    begin_tag = home.find("div", {"name":"col2","class":"col-md-6"})
    begin_tag.append(picture_soup)
    begin_tag.append(text_soup)

    # for data in home(['script']): # used to delete unnecessary tags
    #     data.decompose()

    #
    with open('templates/home_answer.html', 'w') as f:
        f.write(str(home))