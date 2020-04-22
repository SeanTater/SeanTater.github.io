---
layout: post
title:  "Cross compiling Rust Linux Musl binaries on Mac OSX"
date:   2020-04-21 10:23:44 -0400
categories: rust
---
Thanks to
[a brew package by FiloSuttile](https://github.com/FiloSottile/homebrew-musl-cross), and 
[a post by Timryan](https://timryan.org/2018/07/27/cross-compiling-linux-binaries-from-macos.html),
it's actually pretty easy.


```sh
# Install the necessary C cross-compiler
# This takes a while
brew install FiloSottile/musl-cross/musl-cross

# Install the rust target you want
rustup target add x86_64-unknown-linux-musl
```

Now you need to change your Cargo configuration to use your newly installed musl linker
```toml
# ~/.cargo/config
[target.x86_64-unknown-linux-musl]
linker = "x86_64-linux-musl-gcc"
```

Now you should be able to compile your binary
```sh
# Build your app for that target
cargo build --release --target x86_64-unknown-linux-musl

# Your build is in /target/x86_64-unknown-linux-musl/release/your-gizmo
```