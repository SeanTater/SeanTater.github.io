---
layout: post
title:  Strategies for Parallelization
date:   2021-08-26 14:09:00 -0400
categories: rust
---

In this post I am going to give some examples for parallelizing software, using Rust to explain it, but there is nothing special about Rust except that it forbids a few kinds of mistakes. In fact, the best approaches I plan to show in this post was popularized in Go (and to some extent Smalltalk many years ago, maybe others too). It helps that this is a very general problem because it may actually help you understand teams, companies, governments, and other organizations. Most of these strategies could work without code.

# What is parallelization
Parallelism involves performing at least two separate but probably similar tasks with separate tools simultaneously. The analogy to parallelism is that there are sequences of steps to be performed, represented mentally as lines, and they proceed without intersecting. For the same reason, we call these threads. Infinite parallel lines is a bit of an unattainable ideal; in reality they intersect, but you'd like them to intersect as little as possible because that's where the complexity and bottlenecks come in.

In most computer science terminology, parallelism is distinct from concurrency. Parallelism generally means you have separate compute resources (like CPU cores or ALU lanes), but concurrency involves only one compute resource, and the remainder are other resources like storage or network. In principal, you can think of both of these like sequences of steps, but the crucial difference is that with concurrency, the devices can't move forward on their own and they require a CPU (or the like) to give them further instructions, but in parallelism every part is close to autonomous.

I know you're probably bored of reading this groundwork material at this point, but it does matter. If you're coming from a different language, Rust's restrictions are going to seem arcane and arbitrary, maybe even impossible. But they simply enforce things that other languages encourage, and it prevents you from causing yourself a lot of problems you may have missed. In this case, it's Go's documentation we're quoting here and they are experts in this regard.

# Share by communicating, don't communicate by sharing
There are not many problems where data can be processed in parallel without ever sharing anything at all. Suppose for example you want to write a bunch of status messages about your work to a single log file. You can't all open it and write to it simultaneously because it could become garbled if you append, and one message could overwrite another if you don't. So what should you do?

* You can completely ignore the problem and deal with scrambled, unparsable data. Later on when you read it, drop anything that doesn't parse. That doesn't sound that bad until you wonder how you'll know where the next log entry begins. Newlines? Fixed length messages? This is starting not to sound easier, it just delayed the pain.
* You can ensure only one thread has the log file open at a time. Lockfiles could help, shared and distributed files may make it a little more challenging. It's doable. But what about scale? Wait until the lockfile is gone, then create a new lockfile, open a file, write one line, close the file, delete the lockfile, once for every message? That's a lot of file IO and it will end up being a lot of waiting.
* You could give every thread its own log, have them write independently, and merge them later. This is essentially map-reduce, and it scales well without causing garbled results.
* If all the other threads could call one thread to write the logs, then the waiting could happen within the process and avoid the IO, which would be the same problem but still a great speed win.
* What if we could leave messages for a thread to receive asynchronously?

# Enter Channels

Channels are ways to send messages between threads. Like networks, but within a process. Asynchronous message passing has been with us since high performance computing started, but they have been rather challenging to use correctly. (Looking at you, MPI!) That's not true anymore. They are both easy to use and pretty unlikely to cause trouble. They come in a few flavors:

* One thread sends one message to one other thread. Usually called a "one-shot", it's useful for communicating something like "I'm done" and triggering something else to start. Mostly short-lived threads use these. For that reason, they are not super common but if you do use them, you probably use them a lot.
    > In Rust this is typically `tokio::sync::oneshot` or `futures::channel::oneshot`
* One thread sends messages to one other thread. Usually called "spsc" for single-producer, single-consumer, this is a lot like a TCP connection but with separate messages. It only works one way, but it's great for pipelines.
    > In Rust we don't usually use these because `mpsc` can do this just as well already
* Many threads send messages to one thread. (mpsc) This is great for work queues, where the previous stage of compute was parallel, but the next stage may not be parallelizable. This is probably the most common kind of channel.
    > This is super common in part because it's in the standard library: `std::sync::mpsc`
* One thread sends messages to all threads, also called "broadcast", can be useful for some kinds of input data. Be wary of this one because if everyone needs the same copy and only reads it, this might actually be better shared than communicated.
    > I have never needed this because usually I would use `Arc` for this instead of a channel.
* One thread sends messages to any thread, but each message is read only once, also called "mpmc", works like a shared queue. This is one of the most common kinds of channels because queues are so important. It's one of the hardest to write too, but there are good libraries available for them, so don't sweat it.
    > `flume` is the best implementation I know of this in Rust.

# Give channels a shot
We've talked about it quite a bit, let's see what this might look like. First, create a new app the usual way: `cargo new demo_parallel`. Then let's add `crossbeam = "*"` to the bottom of your `Cargo.toml` to fix an issue we'll run into later on. After that's let's play around with them and see if we can learn a bit about what's going on.

```rs
fn main() {
    // Find primes using trial division, an example of a pure CPU problem
    let (sender, receiver) = std::sync::mpsc::channel();
    for block_num in 0..5 {
        // The send half of an mpsc can be cloned, with one given to each thread
        let sender = sender.clone();
        // This thread takes a closure, marked `move` because the thread will take
        // ownership of anything it uses rather than use references
        std::thread::spawn(move || {
            for candidate in (block_num*100)..((block_num+1) * 100) {
                let mut is_prime = true;
                for divisor in 2..candidate {
                    if candidate % divisor == 0 {
                        is_prime = false;
                        break;
                    }
                }
                // This sends a report to the channel
                sender.send((candidate, is_prime)).unwrap();
            }
        });
    }
    // Delete the sender to avoid a deadlock.
    // When all senders are deleted (including the copies each thread has)
    // then the receiver will finish and the for loop will end.
    std::mem::drop(sender);

    // This convenient syntax is available because receiver is iterable.
    for (candidate, is_prime) in receiver {
        println!("{} is {}", candidate, if is_prime {"prime"} else {"composite"});
    }
}
```

# Comparison against the C way

I wanted to start with a good example, but you may have thought of other solutions when you were first looking at this. In particular, you probably thought of using a single shared vector with all the `is_prime` values packed next to each other, like you probably learned in a C class long ago when you first saw this problem.

```rs
fn main() {
    // Find primes using trial division, an example of a pure CPU problem
    let mut is_prime = vec![true; 500];
    for block_num in 0..5 {
        // !! This part won't work. 
        std::thread::spawn(move || {
            for candidate in (block_num*100)..((block_num+1) * 100) {
                for divisor in 2..candidate {
                    if candidate % divisor == 0 {
                        is_prime[candidate] = false;
                        break;
                    }
                }
            }
        });
    }
    println!("{:?}", is_prime);
}
```

But that won't work in Rust, instead, you'll see an error like the following, complaining that you can't move that vector twice. (Remember only one scope can own a variable at a time.) But even if you remove `move` you just get a different error about borrowing.

```
error[E0382]: use of moved value: `is_prime`
  --> src/main.rs:6:28
   |
3  |     let mut is_prime = vec![true; 500];
   |         ------------ move occurs because `is_prime` has type `Vec<bool>`, which does not implement the `Copy` trait
...
6  |         std::thread::spawn(move || {
   |                            ^^^^^^^ value moved into closure here, in previous iteration of loop
...
10 |                         is_prime[candidate] = false;
   |                         -------- use occurs due to use in closure
```

You're running into this because you want multiple threads to edit the same vector at the same time, and Rust forbids that because you can end up with races if the threads somehow wanted to edit the same areas. In this case that won't happen but the compiler doesn't know that. To tell it that, you would `chunks_mut()` instead.

```rs
fn main() {
    // Find primes using trial division, an example of a pure CPU problem
    let mut is_prime : Vec<usize> = (0..500).collect();
    for block in is_prime.chunks_mut(100) {
        // !! This part won't work for a different reason now 
        std::thread::spawn(move || {
            for candidate in &mut block {
                for divisor in 2..*candidate {
                    if *candidate % divisor == 0 {
                        *candidate = false;
                        break;
                    }
                }
            }
        });
    }
    println!("{:?}", is_prime);
}
```

I'm sure you're disappointed to see that didn't work either. It complains about a borrow again, saying that you need to borrow for `'static`, which means forever. You can't do that because then you would never be allowed to read it afterward. That's why `println` fails.

```
error[E0502]: cannot borrow `is_prime` as immutable because it is also borrowed as mutable
  --> src/main.rs:17:22
   |
4  |     for block in is_prime.chunks_mut(100) {
   |                  ------------------------
   |                  |
   |                  mutable borrow occurs here
   |                  argument requires that `is_prime` is borrowed for `'static`
...
17 |     println!("{:?}", is_prime);
   |                      ^^^^^^^^ immutable borrow occurs here
```

In case you're wondering why the compiler wants this, it's because the program instantly moves on to the print before the threads have finished working. We need a way to tell the compiler that the threads are all done and the references are all cleared away. There is a very popular library that does that for us.

```rs

fn main() {
    // Find primes using trial division, an example of a pure CPU problem
    let mut is_prime : Vec<usize> = (0..500).collect();
    crossbeam::scope(|s| {
        for block in is_prime.chunks_mut(100) {
            // This time we will be fine
            s.spawn(move |_| {
                for candidate in block {
                    for divisor in 2..*candidate {
                        if *candidate % divisor == 0 {
                            *candidate = 1;
                            break;
                        }
                    }
                }
            });
        }
    }).unwrap();
    let primes : Vec<usize> = is_prime.into_iter()
        .filter(|x| *x > 1)
        .collect();
    println!("{:?}", primes);
}

```

At last, you have calculated your primes again. All the borrows were cleared when you exited the `crossbeam` scope, which guaranteed that all the threads had quit. Then there were no loans against `is_prime` and you were free to read it again. After that it's smooth sailing.