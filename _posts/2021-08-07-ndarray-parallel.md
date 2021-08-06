---
layout: post
title:  Parallelism in NDArray
date:   2021-08-06 15:00:00 -0400
categories: rust
---

If you're moving from Numpy to NDArray, or even if you use Rust at all, you are probably mindful
of performance and efficiency. So at the risk of preaching to the choir, I would like to
emphasize that high efficiency pays for it's development in several ways that are not always
clearly visible on the outset.
* Pay less for servers, HVAC, electricity, or cloud VMs, quotas and the like
* Scale up to larger problems without any additional orchestration overhead
* Scale down to smaller hardware, or do something on edge devices once done on servers
* Increase reliability of clusters by lowering resource utilization
* Improve responsiveness of live applications

That said, when you have squeezed all the compute power you can from a single core, there comes
a time you need to start looking to distribute. The first place to do that is with multithreading,
and `rayon` is the place to start with that in Rust. Rayon and NDArray go hand in hand because
many of the most compute intensive tasks involve large contigious arrays of numbers, but it's far
from the only way to use Rayon. We might cover other ways in other articles later.

## Basic flavor of Rayon
We already talked about NDArray in another article here. Instead, we'll start with Rayon.
Rayon is a work-stealing queue, with some convenient wrappers around iterators to make creating
queues easy. It's simpler than you might expect, quite dynamic, and usually doesn't require any
tuning. There are obviously downsides and in situations where the work is precisely known, it may
take modestly longer to accomplish the same tasks. But 99% of the time you will not see the
difference, and you will only spend very little code to do it.

If you want to know more about [work stealing][], Wikipedia has a topic on it. But the shortest
description is to imagine that you start by assigning one core all the tasks. Cores with no tasks
left will occasionally check other core's queues, and take the later half of the longest queue.

Using Rayon is about as simple as describing it. In this case a wildcard import is probably
warranted, as most of the features of Rayon are tacked onto existing iterators, and they are only
available if you import them. (There's a reason for this and it's interesting but now is maybe 
not the time to get into it.)

This is an example of how you would use rayon to find the sum of primes from 0 to 100000 using
trial division. Every number from 0 to 100000 can potentially run on different cores, but all
the trials for a single number run on the same core together. This offers a nice balance between
granularoty of scheduling and overhead.

```rs
use rayon::prelude::*;
fn main() {
    let sum_of_primes = (2..100000i64)
        .into_par_iter()
        .map(|i| if (2..i).into_iter().any(|j| i % j == 0) {0} else {i})
        .sum::<i64>();
    println!("Sum of primes: {}", sum_of_primes);
}
```


[work stealing]: https://en.wikipedia.org/wiki/Work_stealing