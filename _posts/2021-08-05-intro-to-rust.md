---
layout: post
title:  Intro to Rust
date:   2021-08-05 13:00:00 -0400
categories: rust
---

Hi friends! There are a number of tutorials that can take you through a basic Rust application,
and in particular I would like to highlight the always-awesome
[Rust Book](https://doc.rust-lang.org/book/) which can take you through tutorials on this and
many other, much deeper and more thought provoking problems!

That said, this post is as much practice for me as it is for you, and I'm looking to target a
few common libraries and common practices over the next few days. It will build on the app we
build here, if you follow along.

## Installing Rust
Check out the main [rust-lang website](https://www.rust-lang.org/) and you will find a good
deal of helpful information for installing and getting acquainted to Rust. Start by installing
`rustup`, which is the tool of choice for maintaining your Rust compiler and related gadgetry.
Just run the following on a Linux or Mac machine and you should be up and running, lickety-split.
It will ask you a number of questions. I recommend going with the defaults, and applying it to
your shell's configuration so that whatever you install will be accessible from within your shell
immediately.

```sh
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Setting up your own app
Now that you have a Rust compiler, the package manager Cargo, and the necessary standard library
and runtime components installed you should be able to get started.

```sh
cargo new yourappname
```

Cargo can create a new application for you from a template, and the default one is the usual
"Hello World". It's quite convenient but it does leave you without any bells and whistles.
In short, you will find the following in your newly created `yourappname` directory:

* A `src/` directory with `main.rs` in it. This is where all of your Rust code needs to stay,
  and anything not in this directory, with a few specific exceptions, will not be built.
* A `Cargo.toml` file that lists all the metadata about your project. You should edit this to
  reflect your new app, and you can add dependencies here when you're ready.
* A `Cargo.lock` file that tracks all the exact versions of all the transitive dependencies
  of your project, so you can have reproducible builds. This is necessary but you don't need to
  read or do anything with it.
* A `.git` directory and `.gitignore` file that constitute a new git repository for your project.
* The `main.rs` file itself, which is a mere three lines, and does exactly what it sounds like.

```rs
fn main() {
    println!("Hello, World!");
}
```

## Starting your new app
Cargo will detect if anything has changed in your source or dependencies since you last built the
app, and it will also automatically build the app before running it if necessary. So as a result,
almost all the time you will find yourself doing the following:

```sh
cargo run
```

And that's it, now you should see `Hello, World!` at the end of the compliation output.
To prove that was your app, you can also run it directly yourself, since it is an actual executable
and does not require any interpreter. Just run this to see your beautiful new app at work.

```sh
target/debug/yournewapp
```

Congratulations!