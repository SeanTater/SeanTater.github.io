---
layout: post
title:  Starting Python with Sentence Transformers
date:   2021-08-09 20:00:00 -0400
categories: python
---

A friend of mine has been learning Python over the past few weeks, since he wants to improve his career opportunities in the future. He made the mistake of telling me about it. It's fun to work through some of the basics again and we took about an hour to start from "Hello World" and end up estimating similarity of phrases at the end. The beginning is easy to understand, but the last step is a bit of a jump I added to make it more interesting.

Here's the overall plan:

What we're doing                | Why we're doing it
--------------------------------|------------------------
Installing Python               | Python will interpret our code and run it
Creating a Virtual Environment  | To install Python packages without admin (optional)
Creating a project              | To organize our code (optional)
Using some inputs               | To interact with users and show it works for any data
Making a vector / list          | To represent a single point in a space
Computing some arithmetic       | To calculate Euclidean distance
Using a loop                    | So you can calculate the distance in any dimension
Using `numpy`                   | So you can calculate the same thing more easily
**Bonus**                       |
Using sentence-transformers     | Represent sentences as vectors in a space
Computing high-D distances      | To represent relatedness of two sentences


> If you're new to this don't freak out of the end doesn't totally make sense. It's making a point that some difficult things still don't take much code. You're not far from amazing results.

# Getting started

Honestly, this is the hardest part. Seriously, it only took us about an hour to write the code and get to results, but installing it on a computer that doesn't have it on it already, especially if you are not familiar with administration like this, could take a while.
**Don't get discouraged if things are slow to get started.**

You need Python installed. I don't use Windows and I don't know much about how you would install it there, just that you can get it at [the Python home page][]. If you have a Mac, it's already installed so you don't have to do anything. If you have Linux, you probably don't need this tutorial and you can teach me instead!

Make sure you can run Python at this point. This means, when you run `python` or `python3` you should see something that looks like the following. Make sure the version starts with `3.` because `2.7` won't suffice. Parts of this will vary depending on the computer you run it on, but that's okay as long as it looks similar.

> On Windows you may need to give the full path to `python3` for this to work

```
Python 3.9.6 (default, Jun 30 2021, 10:22:16) 
[GCC 11.1.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

# Create a virtual environment
I know you're starting with Hello World but I don't want to get stuck in a really limited view of how projects work. You're not alone in the world and millions of people are writing code along with you. You'd be smart to reuse their work, and the way you do that is with libraries. They share their code as libraries, and you use them by installing them and importing them into your code when you want to use them.

Packages are really not much different than applications except that they are not complete on their own and they won't do anything unless they are called as part of an actual app. Think of them like Lego bricks. You can install the packages you need onto your computer but it's a bad idea to do that because there are many versions and you may find you need different packages and different versions for each of your projects. Better to install them into a folder so they can be separate. That folder is called a **virtual environment**. You can create one like so, and feel free to give it a different name.

```sh
# Hash marks here represent comments, you don't have to type them
# The following will create a new environment in a directory
python3 -m venv starter-virtual-environment
#       |  |    |-- Name of the new folder to make
#       |  |-- Python module for making virtual environments
#       |-- Python option meaning to run a package like a script

# Next you need to activate this terminal to use this Python environment
source starter-virtual-environment/bin/activate
# |    |-- This is a relative path to a script that will activate your terminal
# |    |-- Keep in mind the path will depend on what directory you are in
# |-- This means to run a script in the current terminal rather than in an isolated subprocess
# |-- Normally everything runs in separate processes
```

# Set up package management
If you're used to `chocolatey` in Windows, `brew` in Mac, `pacman` or `apt` or `yum` or a dozen others in Linux, then you have used a package manager. The idea is that it will install packages and all the packages they need for you, so you don't need to install them one by one yourself. If you need to upgrade, it can work out the absolute mess, so you can avoid the "DLL Hell" of days of old. These managers are a work of both art and engineering and deserve a moment of silence to recognize the great pain they have saved us all.

*silence*

Okay, so at this point we need to do just that. We want to use Python's package manager `pip` to upgrade itself. Seriously, you can do that and it's totally normal. We do that because we want to make sure that `pip` is ready to install the newest packages. (The formats change a little over time and some packages are only available in new formats.)

```sh
# First, upgrade the package manager in this environment
python3 -m pip install -U pip
# Next install the packages we want, which will also live in this new environment
# This will take a minute
python3 -m pip install sentence-transformers
```

As long as nothing major complained, you should be good. If not, you might get advice from Stack Overflow eiter by searching for a similar problem, or by asking a new question.

# Create a new project
This is easier than it sounds. Python is not very picky and we're not going to write a lot of the metadata for this project because - well - I don't feel like explaining it all and I doubt it would be worth it to you anyway. Also the methods for doing it have gotten a bit complicated for historical reasons.

So all we will do is make two folders and a file.

```sh
# Create a new folder, and a new folder inside it by the same name
# This naming scheme is a bit redundant but it's convention
mkdir -p starterproject/starterproject
# Create a new python script inside your project
touch starterproject/starterproject/main.py
```

Now I'd recommend you open up your new script `main.py` in a text editor or an integrated development environment. I like [VSCode][] and it's pretty easy to set up, but anything, even Notepad, will do. The most important difference at this point is syntax highlighting. It's a little easier to write code that's highlighted, because it gives you visual cues when you make mistakes. In case you are looking for options, there's [Atom][], and some people like [Sublime][] as well but it's not open source. On Linux you probably already have [Kate][] or [Gedit][].

# Hello, World!
This part should be a freebie. You got this.

```py
print("Hello, world!")
```

That's it, just put that in your `main.py` and run it with `python3 starterproject/main.py`

So now let's make it more interesting. Use `input()` to read a single line of text from the terminal. People tend to be confused when nothing happens, so it takes a string that it will print before waiting on the user. Typically, you want to end it with a space, so that their text is not bunched up against yours. Save the output to a variable, like `name` and we'll print it out alongside our greeting. 

```py
# Prints "Hello friend.." then reads until they press enter, the saves the result to name
name = input("Hello friend, what's your name? ")
# Prints "Hello, ", then prints name, then prints a newline (aka enter)
# Most functions don't work this way but print can take any number of arguments
print("Hello, ", name)
```

# Create a vector
For the next few steps it will seem like we lost track but we'll get back to the `input()` we just learned. For now though, wipe out what you have, both the `print()` and `input()` lines. Now we want to use a `list`, which is a series of objects, represented by enclosing square brackets. Some languages call them `vector` already and in CS they would be called `array`. Don't be too concerned about terminology, for the purposes of our tutorial today they're all the same. Whatever you want to call it, this is what we're talking about:

```py
vec1 = [1, 2]
```

It's a list with two elements, and they're both integers. At this point it's worth mentioning all the types you'll see in Python here, since now you have seen several. There are not that many basic types and you have seen most of them. The cool part is that you can make new types yourself by composing them. It's easy but we're not going over it today.

The type's name | An example |  What it does
----------------|------------|----------------------------------
`str`           | `"Hello"`  | Stores any number of characters in sequence.
`int`           | `5`        | Stores any integer exactly, positive or negative.
`float`         | `6.7`      | Stores any real number, to about 14 significant digits.
`list`          |`["hi", 5]` | Stores any number of objects in sequence.
`dict`          | `{5: -1}`  | Relates objects to each other. Here, `5` maps to `-1`
`function`      | `print`    | A self-contained block of code, possibly with inputs and outputs
`class`         |            | We haven't gotten there yet. This is how you make new types.
`np.array`      |            | A specific new type we want to use later in the tutorial.

# Doing some arithmetic
Computers obviously do more than store and relay information, and we need to do a little calculation in this case. We do that one number at a time (which we would call a `scalar`, to borrow another math term) and that's super easy to do. We're going to compute the Euclidean distance between two points here.

> Remember, euclidean distance is the square root of the sum of squared error.
> It may help to think of it as Pythagora's theorem, but arranged so that we can find `c`
> rather than `c**2`

```py
point1 = [1, 2]
point2 = [0, 5]

dist =        (point1[0] - point2[0]) ** 2
#                    |                |-- This is exponentiation.
#                    |                |-- This squares the difference on the left
#                    |-- We select an element from a list using square brackets.
#                    |-- They count from 0 for historical reasons, so this is the first element. 
dist = dist + (point1[1] - point2[1]) ** 2
# We now also need to calulate the squared difference for the second dimension
dist = dist ** 0.5
# And finally, we find the square root, which is the same as raising to the half power

print(dist)
# For fun, print it out
```

There are several new parts in this block, which we should take a moment to break down:
* You can get any element from a list using a syntax that looks a lot like another list:
  * If your list looks like `a = ["x","y","z"]` then `a[1]` is `"y"` because it counts from zero.
* You can raise numbers to any power, even negative or fractional, using `base ** exponent`
* You can use a variable in defining itself, as long as it's not the first time it's mentioned.
  * Notice how there is `dist +` in the second line, but it is not in the first line.  

> Take a moment to open up a Python terminal, using `python3` and not passing any script name,
> so that you can work through these parts and understand each part individually.
> About half the code I write is into terminals like these on the spot, so if you're ever not sure
> how something will work, go ahead and try it. 

# Using a loop
Finding the distance between points may be neat but it's not super flexible. It doesn't look bad for a two dimensional example, but we're going to shoot for more than that. Not just two or three, but even three hundred or three thousand dimensions. **You can do that** and it's not that hard. Here's how:

```py
# We're just upping it to three dimensions for starters
point1 = [9, -1, 1.2]
point2 = [0, 0, -1]

dist = 0

for component in range(3):
    # Every line inside this loop will repeat.
    # First with coordinate = 0, then with coordinate = 1, then coordinate = 2
    # It will loop a total of three times, and coordinate will not reach 3
    dist = dist + (point1[component] - point2[component]) ** 2

dist = dist ** 0.5
```

Simple enough? It just counts from 0 to one less than the number you set (in this case three).
You'd still need to set the number of dimensions you want, but we're really close.

```py
# For grins and chuckles check out what happens if you do nothing with the items from `range()`.
# You can confirm it's just plain old numbers from 0 to 2, one at a time
for index in range(3):
    print(index)

# Range is the easiest way to do it but for works for anything with multiple contents.
# This will print elements from a list one at a time.
# If you used [0, 1, 2] you would get the same result as what range gave you.
for element in ["cat", 5.1, "dog"]:
    print(element)

# This prints "a", then "s", and so on one at a time, because strings are iterable too
for letter in "astronaut":
    print(letter)

# There are a number of functions to do interesting things with iterables but we only need one
for firstname, lastname in zip(["Joe", "John", "Sue"], ["Zhou", "Johnson", "San"]):
    print("Hello ", firstname, lastname)
```

Now let's use what we learned to improve the loop. We'll eliminate the need to write in the number
of times we want the loop to repeat, and we'll also go through both vectors at once. So, one more time, from the top:

```py
point1 = [9, -1, 1.2]
point2 = [0, 0, -1]

dist = 0

for component1, component2 in zip(point1, point2):
    # This is about as easy as this line can get, it's very much like the definition.
    dist = dist + (component1 - component2) ** 2

dist = dist ** 0.5
```

# Going all in with the automation
At first, we did every scalar by hand. Then we showed the interpreter how to do one scalar and had it repeat for every dimension. Now we will level up one more time. It still does the same thing, but it's an even more succinct way to say the same thing. The first two ways were for your understanding but this way is how you would normally do it in industry. Guess what? We're going to start again from the top.

```py
# Numpy is a library. You have to import it first to use it.
# In this case I renamed it to `np` to save typing.
import numpy as np

# Numpy wraps lists to make them into what they call arrays
# We have to wrap each list we want to treat this way
point1 = np.array([9, -1, 1.2])
point2 = np.array([0, 0, -1])

# Numpy will allow us to treat the whole array as if it was a scalar
# So we can do elementwise arithmetic on the whole thing at once like this.
# Essentially, it wrote the `for` loop for you.
dist = (point1 - point2) ** 2
# It also gives us a bunch of functions that work on arrays, like this finds the sum.
# When it does, the result is a scalar and we can take the square root as normal
dist = np.sum(dist) ** 0.5

# Now we have the distance between those two points and everything from here on proceeds as normal
print(dist)
```

Numpy is not magic. It's the same things you would have written yourself, if you had the time. It's just written by experts, and for the most part in C, so it can be much more tuned for your hardware. As a result, it's not just easier, it's faster. It's often even hundreds of times faster. So take a moment to play with it; if you work in data analytics you'll probably reap the rewards of your practice many times over.

# Bonus round: find sentence relatedness
So far we started from the ground up, although we did move pretty fast. But I want to show you one more part. You're really close to something useful, which might change your expectations of what you can do.
We're essentially going to use a thesaurus, but it's much cooler because rather than working on words, it works on whole phrases or sentences.

```py
import sentence_transformers as st
# This model is trained on gigantic amounts of data at universities and large tech companies
# and we are merely downloading the results of their massive computation, so we can use them
# on our own data.
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Remember we said the part from Hello World would be making a comeback
message1 = input("What's your first message: ")
message2 = input("What's your second message: ")

# Now this part is the real magic
# To be serious it is quite complicated and I admit I don't fully grasp it all either
# But it represents every phrase as a vector, and phrases with similar meanings will be closer
# together than those that don't. Luckily, it's very easy to use.
point1 = model.encode(message1)
point2 = model.encode(message2)

# I bet you remember this part!
dist = (point1 - point2) ** 2
dist = np.sum(dist) ** 0.5

# That's it, this is our estimate of similarity
print(dist)
```

Try it out a few times, run it with `python3 starterproject/main.py` and give it some phrases.
Similar phrases should have lower numbers, and different ones should have higher numbers.

# Extra Credit
If you check out [the documentation for sentence transformers][], you can see the right metric is cosine similarity for these models, not euclidean distance. The results will be similar in that they have the same smaller-is-more-related property, but cosine similarity is more accurate. Try using their utility methods for it and see if you can work out how to do it yourself. It should take only about 5 lines if you use numpy.


[the Python home page]: python.org
[VSCode]: https://code.visualstudio.com/
[Atom]: https://atom.io/
[Sublime]: https://www.sublimetext.com/
[Kate]: https://apps.kde.org/kate/
[Gedit]: https://wiki.gnome.org/Apps/Gedit
[the documentation for sentence transformers]: https://www.sbert.net/docs/pretrained_models.html