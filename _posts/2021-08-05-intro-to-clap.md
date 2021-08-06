---
layout: post
title:  Intro to Clap
date:   2021-08-05 13:30:00 -0400
categories: rust
---

Most Rust applications will sooner or later need to take some command line arguments, even if
they are GUI apps, and possibly even web servers or microservices. There are plenty of ways to
do this, and while `clap` is not necessarily the simplist or applicable to everyone, it is quite
flexible and appropriate for most circumstances. It's also the most common, from my small and
definitely unscientific survey of applications.

> If you are familiar with Python, you will probably find clap to be very similar to argparse.

## Getting started
To run through this tutorial, you need to have Rust and Cargo installed and you need an app
that you want to add clap to. We just created it with `cargo new tryrust` but you may want to add
it to an existing app. Clap is almost independent of the rest of your application unless you want
to get into subcommands, which are admittedly very cool.

So at this point you should have a `Cargo.toml` that looks something like 

```toml
[package]
name = "tryrust"
version = "0.1.0"
edition = "2018"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
```

You should have a `main.py` that looks like:

```rs
fn main() {
    println!("Hello, World!");
}
```


## Add Clap
Adding clap to the dependencies is really easy, you just tack it onto the dependencies list.
It helps if you know what version you're looking for, and eventually as this post ages you may need
to upgrade to a newer version than shown here.

```toml
[package]
name = "tryrust"
version = "0.1.0"
edition = "2018"

# See more keys and their definitions at https://doc.rust-lang.org/cargo/reference/manifest.html

[dependencies]
clap = "2"
```


Clap offers a number of ways to specify arguments, some more comprehensive, and others more
succinct. While it may be convenient to use the succinct "usage" style and YAML based approaches
I still prefer the object oriented approach, in part because it seems to me like it involves the
least magic, and also because my IDE supports it nicely. This is what it looks like for about the
simplest use case you might have for it.

```rs
fn main() {
    let args = clap::App::new("Example greeting app")
        .arg(clap::Arg::with_name("name")
            .takes_value(true)
            .required(true)
        )
        .get_matches();
    let name = args.value_of("name").unwrap();
    println!("Hello, {}!", name);
}
```

If you are familiar with `argparse` in Python, you will probably be very comfortable with `clap`.

* The syntax `clap` parses is the same as `argparse` in every way I can find
* `clap` does not enforce types, and it's up to you to parse the strings yourself
* `Arg` is an object with quite a few methods, rather than a method that takes keyword arguments, 
  because Rust doesn't support keyword arguments.
* The resulting arguments are `Option<&str>` because the argument may not have been given, and
  if you want to have a default, you can either use `args.value_of(...).unwrap_or("some default")`
  or you can use `clap::Arg::with_name("...").default_value("some default")` when defining the
  Arg.

## More advanced clap
Clap supports quite a few interesting features, spanning most of what you would realistically
want to do with a command line interface. (There are some exceptions I can think of, like
recreating `ffmpeg`'s order-sensitive parsing, but even humans can find that difficult to track.)

The most interesting feature I would want to cover is subcommand parsing, because while it does
sound a bit niche at first, it makes a lot of sense for commands that do different but related
tasks. Take for example, how `xz` might have made sense more as `xz compress` and `xz decompress`
instead of taking options. You could have implemented this as a required argument, where only
two strings are allowed, but then that means you would have a lot of options mean nothing in the
other context. What would `xz decompress -9` mean? Subcommands solve this by giving options a
context, and only defining arguments in the scope they have meaning. It's obvious in hindsight,
but when I first started using them I found it quite impressive.

### The option-based method

> You can find the [API documentation for clap][] on doc.rs

```rs
fn main() {
    let args = clap::App::new("Fake compression app")
        .arg(clap::Arg::with_name("input_file").takes_value(true).required(true))
        .arg(clap::Arg::with_name("output_file").takes_value(true).required(true))
        .arg(clap::Arg::with_name("decompress").short("d"))
        .get_matches();
    let name = args.value_of("input_file").unwrap();
    let decompress = args.is_present("decompress");
    if decompress {
        println!("Decompressing {}", name);
    } else {
        println!("Compressing {}", name);
    }
}
```

`decompress` is a boolean here, and you probably end up calling two different procedures that
do almost unrelated tasks at that point. You'll either need to go ahead and handle all the matched
args at that point with many `value_of()`s or else you could pass the ArgMatches and handle them
later.

### The mode-based method

```rs
fn main() {
    let args = clap::App::new("Fake compression app")
        .arg(clap::Arg::with_name("mode")
            .takes_value(true)
            .required(true)
            .possible_values(&["compress", "decompress"])
        )
        .arg(clap::Arg::with_name("input_file").takes_value(true).required(true))
        .arg(clap::Arg::with_name("output_file").takes_value(true).required(true))
        .get_matches();
    let name = args.value_of("input_file").unwrap();
    let decompress = args.is_present("decompress");
    match args.value_of("mode").unwrap() {
        "decompress" => println!("Decompressing {}", name),
        "compress" => println!("Compressing {}", name),
        _ => unreachable!("Can't happen, clap would have exited the program")
    }
}
```


### The subcommand-based method

Notice this method is significantly longer than the other two, but it's because I added a few
details to highlight what you can do.
* You can have application arguments that every command requires
* You can also have arguments specific to any command, exactly the same as the full app
* When you use a subcommand, it has a set of matches just like the parent app did,
  which you can retrieve when you `match` against `args.subcommand()`

```rs
fn main() {
    let args = clap::App::new("Fake compression app")
        .arg(clap::Arg::with_name("mode")
            .takes_value(true)
            .required(true)
            .possible_values(&["compress", "decompress"])
        )
        .arg(clap::Arg::with_name("input_file").takes_value(true).required(true))
        .arg(clap::Arg::with_name("output_file").takes_value(true))
        .arg(clap::Arg::with_name("verbose").short("v"))
        .subcommand(clap::App::new("compress")
            .arg(clap::Arg::with_name("quality").short("q")
                .takes_value(true)
                .default_value("6")
                .possible_values(&["1","2","3","4","5","6","7"])
            )
        )
        .subcommand(clap::App::new("decompress")
            .arg(clap::Arg::with_name("block-parallel").short("p"))
        )
        .get_matches();
    let name = args.value_of("input_file").unwrap();
    let decompress = args.is_present("decompress");
    match args.subcommand() {
        ("decompress", Some(sub_args)) => {
            println!(
                "Decompressing {}, {}",
                name,
                if sub_args.is_present("block-parallel") {"in parallel"} else {"in serial"}
            )
        }
        ("compress", Some(sub_args)) => {
            println!(
                "Compressing {}, with quality {}",
                name,
                sub_args.value_of("quality").unwrap().parse::<isize>().unwrap().
            )
        }
        _ => println!("Please choose a subcommand")
    }
}
```

## Conclusion
Clap is one of those libraries that are so frequently used that it could almost be part of the
standard library, but since the standard library has very strict backward compatibility standards,
it will probably never be added. Instead, spend the extra line in your `Cargo.toml` to add it to
your projects when you start a new one. Even if in the first few days you don't think you'll need
it, I bet as your project matures, you'll use it later.


[API documentation for clap]: https://docs.rs/clap/2.33.3/clap/index.html