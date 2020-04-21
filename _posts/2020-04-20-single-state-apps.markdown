---
layout: post
title:  "Single State Web Apps"
date:   2020-04-20 17:23:44 -0400
categories: web
---

Build straightforward ML web apps using websockets that can still move to production

# Intro
Most data scientists consider web development something of a nuisance. It's a pity, because making
your point to nontechnical folks often hinges on making a good impression that you know what you're
doing, and visual presentation is a big part of that.

We know that a lot of data science software doesn't make it to production, and in my case I think
the ratio is probably at least 5:1. So porting all of the good practices of production software to
data science just won't fly because people don't want to waste that kind of time and inflate their
time to MVP. Instead, we need to figure out ways to keep from boxing ourselves in too much so that
our software can't adapt to its new goals if it should survive.

There are a lot but today I want to go over one of the most interesting ones, which I called a
single-state application, but you might call it by something else, if so, drop me a line. I'd like
to hear about it.

# The Problem
[Streamlit](https://www.streamlit.io/) apps are just the bee's knees at this point in DS from my
experience and it certainly delivers on being the fastest imaginable (and most opinionated) way to
build an application ever conceived. This Jekyll app (the blog is based on) took me longer.

But the trouble is that some applications need to interact with the user as you build the model,
and it may not proceed in such a linear fashion. Then you'll probably have to eject from your
Streamlit world and strike it out on your own. But don't despair! You can still build something
interactive in under an hour, so while it's slower, if you found this page, you're halfway there.

# The Concept
Consider the now-standard web app, with an interactive Javascript frontend and an API server as a
backend. They are well understood, operate by REST calls, and have pretty good standards. The issue
with using them for highly interactive workloads is that they typically get hogtied by the
database. Consider that your application looks like this

```
{Browser}*N -> Load Balancer -> {App Server}*M -> Database
```

Every browser can connect to different app servers, which is what allows you to scale. But each
request from the same browser could go to different app servers. (It's possible with hashing that
they will all go to the same one, but this is not typically something you can rely on.) As a
result, all persistent state for that user has to make constant round-trips to the database, which
in the case of our huge data sets is already beleaguered.

So let's break that up. Wouldn't it be nice if every app server could uniquely own a specific
user's session? It can! You can do that with *websockets*. On the face of it, it looks something
like this:
```
Browser X -> Load Balancer -> App Server X -> Database
```
But actually it's a lot more like the following, but it doesn't really matter in practice, as long
you do your best to *write only non-blocking async code*.
```
Browser X of N -> Load Balancer -> App Server (hash(X) mod M), coroutine (X // M)  -> Database
```

The advantages are many.
* Semi-ephemeral working set data doesn't need to be transferred
* Most serialization latencies are gone
* Network traffic is reduced
* Most failures only affect one user
* It's just so darn easy!

# Implementation

> You can find a working demo on Github under [SeanTater/single-state-app](https://github.com/SeanTater/single-state-app)

For the sake of this demo, I made a few less common design decisions. I'm using Preact+HTM, which
is great for quick loading applications that don't have too many dependencies. I use it so that
you don't need to install npm and hundreds of modules to do something simple. You will still need
Python 3.6 or greater and the websockets module though.

## The backend
The backend is centered on a single class and it's methods. Yours will be different, but it will
probably have the same _send, assign, and route methods:

```py
# app.py
class Router:
    def __init__(self, ws):
        ...
    
    async def _send(self, **message):
        await self._ws.send(json.dumps(message))
    
    async def assign(self, key, value):
        ...
    
    async def axpy(self, a, x, y):
        ...
    
    async def route(self):
        """ Start an event loop directing messages to a methods on Router according to their tag.
        """
        ...
```

The entry point of the application just starts an endless event loop reading and writing to a
websocket. It uses a small lambda wrapper around Router to kick off each connection.

```py
# app.py
if __name__ == "__main__":
    start_server = websockets.serve(lambda ws, _path: Router(ws).route(), "0.0.0.0", 8001)
    asyncio.get_event_loop().run_until_complete(start_server)
    asyncio.get_event_loop().run_forever()
```

Once that's the messages are passed to the router, which along with some exception housekeeping,
calls the method named by "tag" in the incoming message.

```py
# app.py
tag = message.pop("tag") # Remove and use the tag
await getattr(self, tag, lambda **_x: print("Route not found:", x))(**message)
```

In response, we can send any messages we want, multiple or 0 and all different times and types.
We even draw progressbars this way; quite convenient!

```py
# app.py
# Compute a * x + y and send a message names "assign" to set a key "result_vector" in the browser
await self._send(tag="assign", key="result_vector", value=[a * xi + yi for xi, yi in zip(x, y)])
```

## The frontend

The frontend certainly has a bit more code, but it's still not very complicated. The entry point
is naturally `index.html` but it quickly passes the buck to `index.js`, which promptly replaces
the entire document body with the output of the App function near the bottom:

```js
// index.js
const App = () => {
    const [session, publish] = openSession();

    // This is not quite JSX but it's close and it's very minimal
    return html`
        <${Session.Provider} value=${[session, publish]}>
            <${TrivialComponent} />
            <${BasicSessionComponent} />
        <//>
    `
}

render(html`<${App} />`, document.body)
```

When we opened a session in App(), we
did several things:
```js
// session.js
// We opened a new websocket connecting to the backend
const SOCKET = new WebSocket(`ws://${window.location.hostname}:8001/ws`)
```
```js
// session.js
// We set up a reducer, which will handle all incoming messages, similar to the entire Router class
// In most cases the frontend reducer shouldn't be super complex compared with the UI components
// anyway, so it tends not to be that big of a problem. YMMV.
const reducer = (state, action) => {
    switch (action.tag) {
        case "assign": return {...state, [action.key]: action.value}
        case "error":  return {...state, error: action}
        
        // Not everything has to be shared; in this case we choose some messages should propagate
        // to the server and some not. This is important to prevent loops, so keep that in mind.
        case "axpy":
        case "any other messages you want..":
            // This line forwards anything back to the server
            SOCKET.send(JSON.stringify(action));
            return state;
        
        // No change; you could also choose to send to the server here.
        default: return state
    }
}
```
```js
// index.js
const App = () => {
    // "session" is the current state of the frontend,
    // and "publish" is a callback to sends a messages to the reducer,
    // and in turn only the reducer can change "session"
    const [session, publish] = openSession();

    // Session.Provider passes both values down the tree of components
    return html`
        <${Session.Provider} value=${[session, publish]}>
            <${TrivialComponent} />
            <${BasicSessionComponent} />
        <//>
    `
}
```

Lastly, we need to actually use the new values that the backend gave us in our final component:
```js
// index.js
const BasicSessionComponent = () => {
    // Retrieve the receiver, sender pair broadcast by the Provider
    const [session, publish] = useContext(Session)

    // Some parts were left out for clarity. Get the full code from the repo.
    return html`
        <article class="card">
            <header> Multiply Vectors </header>
            <p> Compute a * x + y </p>
            <button onClick=${publish({tag: "axpy", a, x, y})}> Run Calculation! </button>
            <p>Result: ${session.result_vector.join(" ")}</p>
        </article>`
}
```

And that's it! You have an interactive websocket app!