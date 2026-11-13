# Qualitative samples (10 evaluation posts)

Each post shows the human reference and the greedy SFT, PPO and DPO summaries with their rφ score and ROUGE-L. The *Why* line is the author's judgement after reading the post.

## Sample 1 — eval post #50 (candidate: PPO/DPO better (ROUGE-L))

```
SUBREDDIT: r/offmychest
TITLE: someone stole my Jeep Wrangler, and someone wrecked into my parked car
POST: Some scumbag stole my Jeep Wrangler on Friday. It was recovered, stripped and destroyed, in the ghetto of the local major city before I even knew it went missing. After filing some reports and making a few calls I determined that I couldn't go get it any time soon, so I went camping for the weekend - as previously planned. 

When I came back home Sunday, I parked my car in it's usual spot at my apartment and went inside to clean up, drop off gear and get ready to leave - I needed to find the parts to fix what I could, on my Jeep; hopefully I could make it drivable. Instead, about twenty minutes into my routine, my neighbor knocked on my door to notify me that some ass hat wrecked into my parked car while I was inside.

I had to take a day off of work yesterday to sort through this mess. I found the Jeep at the impound lot and discovered that the wheels and tires were stolen, the brand-new catalytic converter was cut off, they ripped out the door lock (instead of unzipping or cutting through the soft top), they sheared through the security console, broke the steering column and ripped out the ignition - after cramming a screwdriver into all of the external locks, permanently fucking them all up. Everything inside was stolen, except for my tent rainfly, which wasn't a whole lot but it still adds to the indignation. I had to buy a complete set of wheels and tires just to drive it out of impound, and wasted several hundred more dollars paying for the tow and impound fees. Also, I'd recently removed full coverage since the Jeep was just sitting around as a leisure vehicle and I hadn't driven it much in the last year or so. So, I get to eat the several thousand dollar cost for the theft. Also, most of the cops I had to deal with were really condescending.
TL;DR:
```

- **Reference:** Scumbag stole my Jeep Wrangler on Friday, and some goomba wrecked into my parked car on Sunday; most of the cops were very condescending.
- **SFT** (rφ -1.12, ROUGE-L 0.169, 43 tok): I had to pay for a whole lot of cops to get my Jeep out of impound, and I had to pay for a whole lot of cops to get my tent out of impound.
- **PPO** (rφ 0.49, ROUGE-L 0.578, 35 tok): Someone stole my Jeep Wrangler, wrecked into my parked car, and ruined the entire van. Also, the cops were really condescending.
- **DPO** (rφ 0.14, ROUGE-L 0.316, 45 tok): Someone stole my Jeep Wrangler, wrecked into my parked car, and ruined my camping trip. I had to buy new tires and wheel covers, and had to pay for tow and impound fees.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 2 — eval post #18 (candidate: PPO/DPO better (ROUGE-L))

```
SUBREDDIT: r/AskReddit
TITLE: What should I do with my schizophrenic Grandfather
POST: As the title says, i'm not to sure what I should do with him. Im 17, and I take care of him while my mom is at work. Usually that's just making sure he takes his medication, and eats food. Recently, hes started to drink. His psychiatrist ( amazing lady ) told him he has to stop, and he agreed to, but he hasn't. My mom thinks that all the booze is gone, but I know he still has some ( called the local taxi company to pick him up some vodka ). Last  night, he wouldn't take his meds, and drank instead. 

I'm not sure if I should tell my mom about this, it is quite a common thing when she isn't here. The only thing that is making me not tell her is that he is 72, it is his life. He is well aware he is ill, and knows drinking isn't a good combo with his meds.
TL;DR:
```

- **Reference:** Sxhizophrenic grandfather is drinking/refusing to take meds. Should I leave it alone, or tell my mom?
- **SFT** (rφ -2.08, ROUGE-L 0.205, 30 tok): Grandfather is sick, and I don't know what to do. He is 72, and I don't know what to do.
- **PPO** (rφ -0.95, ROUGE-L 0.556, 26 tok): Grandfather has booze in his system, but refuses to take his meds. Should I tell my mom about this?
- **DPO** (rφ -1.02, ROUGE-L 0.233, 33 tok): Grandfather has started drinking, and is not taking his meds. I don't know what to do with him. He is well aware he is ill.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 3 — eval post #271 (candidate: PPO/DPO better (ROUGE-L))

```
SUBREDDIT: r/relationships
TITLE: Me [19M] meeting my 1/2 Brother [~15] for the first time
POST: **Backstory**

My dad left ~16 years ago. I have never meet him. He was a raging alcoholic and cocaine addict when he left and seems to have cleaned up his act. My grandma may have cancer so my mom contacted my dad so they could have their goodbyes should worse come to worse and my grandma die suddenly. Turns out I have a half brother whom we'll call John. I want to meet John but I don't want to meet my dad. 

One of my 2 sisters has already met with my dad and the other is planning to this weekend. I have no intentions of meeting him. They both have John's number and didn't give it to me. I haven't asked for it but I didn't know they had it until this morning when my mom let it slip by mistake.

I plan on asking for John's number and calling him. I assume he know's I exist because my sister's have his number and I'm assuming they have contacted him. I just don't know what I would say to him so start a conversation. I know I should start out with the, "Hi, John. I'm Mickey_Done, your long lost brother. How are you?" I just don't know where to go from there.
TL;DR:
```

- **Reference:** What should I say to John when I call him, past, "Hi, John. I'm Mickey_Done, your long lost brother. How are you?"
- **SFT** (rφ -1.09, ROUGE-L 0.182, 25 tok): I'm meeting my 1/2 Brother for the first time. I don't know how to start a conversation.
- **PPO** (rφ -0.28, ROUGE-L 0.508, 48 tok): Meeting my 1/2 Brother for the first time. Don't know where to start a conversation. Should I start with the, "Hi, John. I'm Mickey_Done, your long lost brother. How are you?"
- **DPO** (rφ -0.30, ROUGE-L 0.157, 33 tok): My dad left ~16 years ago. I have no intentions of meeting him but I want to meet my brother. I don't know where to start.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 4 — eval post #12 (candidate: PPO/DPO worse (ROUGE-L))

```
SUBREDDIT: r/relationships
TITLE: Me [00 M/F] with my ___ [00 M/F] duration, short-description
POST: New here, but you all have come highly recommended from a close friend and I appreciate any comments, thank you. Here it is.

We both go to college together, same major, and we sit next to each other in two classes. We started hanging out Oct 5th of last year and it was great. Shes smarter than me and helps with homework which is nice, but its not a reason to keep a relationship alive.

There are two main problems that I've been dwelling on for the past few weeks. As far as I can tell, she is very much in love with me, but I've come to see that although I like her, I do not love her. One problem remains with this because I cant determine a definite reason why I don't...

I don't know how to break up with her. I care about her, so I don't want to crush her, but I think its just as bad to go on thinking that, "maybe if I stick it out for awhile I will see something different and then love her again."

Best case scenario: We remain friends and can both look fondly on memories shared, mostly that she can move on without excessive agony. I'd rather no longer be friends and have her be happy sooner than try to hold up a friendship if you guys think it'll make a difference.

Any help or comments are greatly appreciated and questions are also welcome! Thank you so much for reading.

Excessive background: Small classes (20+-5). Military with similar interest in field so good chance our paths will cross for at least the next two years. This is my longest relationship I've ever held. She just had her wisdom teeth out today. Im leaving for a football game at noon tomorrow.
TL;DR:
```

- **Reference:** Almost 1 year, intertwined lives, she loves but I don't. I don't know when or how best to break it off.
- **SFT** (rφ -1.15, ROUGE-L 0.512, 24 tok): I like my girlfriend, but I don't love her. I don't know how to break up with her.
- **PPO** (rφ -0.63, ROUGE-L 0.111, 38 tok): Long-term friend and girlfriend of 2 years, but both in love with each other. Both in love with her but not in love with me. Any advice would be great.
- **DPO** (rφ 0.04, ROUGE-L 0.310, 42 tok): Long relationship with friend, both have feelings for each other, but I don't love her. I don't want to crush her, but I do love her. How do I break up with her?

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 5 — eval post #34 (candidate: PPO/DPO worse (ROUGE-L))

```
SUBREDDIT: r/AskReddit
TITLE: If a person does certain things with one hand and certain things with the other, but cannot do these things with the opposite hand, what are they considered? More detailed and coherent question inside.
POST: First off I'll say the obvious. The reason I am asking this is because it describes me.

I write with my left hand. I brush my teeth with my left hand. If I'm using a utensil to eat, I use my left hand. But if I just grab food, I use my right. I play sports with my right. If I were asked to raise my hand, I would raise my right. I play double bass (string bass) and guitar and play both right-handed.

I guess I might as well add that I'm right-footed. Unfortunately I can't write with my left foot.

I've started taking an intro to psychology class and am learning more about the right and left sides of the brain. I don't understand why, instead of favoring one side of the brain for everything, my body favors a different side depending on the activity. I was just curious if anyone has any knowledge about this sort of thing. I find the brain extremely interesting and wanna know what the heck is going on with mine.
TL;DR:
```

- **Reference:** I'm not right-handed. I'm not left-handed. And I'm not ambidextrous. What does that make me?
- **SFT** (rφ -2.13, ROUGE-L 0.488, 28 tok): I'm right-footed and I'm left-handed. I'm not sure if I'm right-footed or left-handed.
- **PPO** (rφ 1.06, ROUGE-L 0.111, 39 tok): If a person does certain things with one hand and certain things with the other, but cannot do these things with the opposite hand, what are they considered? More detailed and coherent question inside.
- **DPO** (rφ 1.06, ROUGE-L 0.111, 39 tok): If a person does certain things with one hand and certain things with the other, but cannot do these things with the opposite hand, what are they considered? More detailed and coherent question inside.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 6 — eval post #60 (candidate: PPO/DPO worse (ROUGE-L))

```
SUBREDDIT: r/relationships
TITLE: My (M/24) girlfriend (F/23) is trying to convert me into a Muslim, would like some insight on this.
POST: Hey /r/Relationships! This is an obvious throwaway account. I'll try to post as much detail as possible.

Some background, when I was little, my parents raised me as a Catholic. But as I grew older, I did not associate myself with any religion. I am **not** against any religions, in fact, I have a very open view on all religions and I appreciate everyone's opinions and faith.

Today, my girlfriend asked me to convert into a Muslim. At first I thought it was a joke, because her and her friends would always joke about me becoming a Muslim. But I found out she was being serious and I wasn't sure how to respond but to say "I can't."

I also want to say that before we started dating, we made an agreement that I will not convert and if that is something that will get in between us, I don't think a relationship would be a good idea.

Anyways, she goes on and on about why I should. So I asked her, why all of a sudden she's asking me to convert. She says that before we get any further into our relationship, her parents wants me to convert.

Now, I really love this girl, but I don't want to convert for the wrong reason. I keep telling her that if I converted, it'd be for the wrong reason. (I've been told that converting for your wife is not a good idea. Something along those lines.)

So what do I do? Is there anything I can do to compromise?
TL;DR:
```

- **Reference:** Girlfriend/Girlfriend's Family wants me to convert into a Muslim. I don't want to. Any way I can compromise?
- **SFT** (rφ -0.10, ROUGE-L 0.636, 28 tok): Girlfriend wants me to convert into a Muslim, I don't want to, but I don't want to be a bad person.
- **PPO** (rφ 0.79, ROUGE-L 0.383, 31 tok): Girlfriend asks me to convert into a Muslim, would like some insight on this. Would like to compromise but would like to know what to do.
- **DPO** (rφ 0.80, ROUGE-L 0.314, 35 tok): Girlfriend is trying to convert me into a Muslim, would like some insight on why she's asking me to do so. Would like some advice on how to compromise.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 7 — eval post #62 (random)

```
SUBREDDIT: r/relationships
TITLE: I [27 M] have "almost arguments" with my wife [28 F] about cleanliness of things and related topics. Reddit, am I just a slob, or is she being unreasonable or (medically) OCD? Looking for opinions.
POST: So we've been married for quite a while, and get along famously on nearly everything. Except for one category, and that's the general sanitation level of things.

We both like to keep an orderly house, but specifically I mean things like sitting on a bed after having worn pants outside and sat down places. One time she acted like I was crazy for putting some jeans I had worn a few times (just out and about) back into a dresser drawer that I use for gym shorts and mostly clean things.

It's this among other things, with more examples I can give if necessary. The reason this bugs me from a relationships point of view is that while so far I've tried to just be glad that she's clean and tidy instead of falling to the other side of things and picking my battles elsewhere, this wears on me when through shock and surprise she kind of makes me feel like I was raised in a barn.

I am truly not sure whether I am in the wrong or is she on some of these things though. I don't possess the exact biological knowledge to say whether there's is reason or rhyme to this type of thing, but in general I just don't stress over the bottom of a piece of luggage that truly makes me wrong for putting it on the bed. I don't know. Or if it's normal to feel the need to run a bleach only cycle to "clean out" the washing machine after a load of soiled laundry went through previously.
TL;DR:
```

- **Reference:** Based on the above examples, am I the dirty one or is she being a little obsessive? Is it worth confronting her about or just don't sweat the small stuff?
- **SFT** (rφ -1.32, ROUGE-L 0.194, 33 tok): I have a wife who is OCD and I don't know if I am in the right to feel the need to clean things that I don't want to.
- **PPO** (rφ -0.22, ROUGE-L 0.076, 48 tok): My wife makes me feel like I was raised in a barn when she makes me feel like I was raised in a barn when she makes me feel like I was raised in a barn when she makes me feel like I was raised in a barn
- **DPO** (rφ 0.02, ROUGE-L 0.137, 45 tok): My wife makes me feel like I was raised in a barn when things like sitting on a bed after wearing pants outside and sitting down places, and I don't know whether I am in the wrong for feeling this way.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 8 — eval post #13 (random)

```
SUBREDDIT: r/relationships
TITLE: I [17 m] need help
POST: I don't know what to do. I feel like while I'm good looking I can never find any one who cares enough to get to know me for me. I have 5 best friends that are girls, I had a thing with one, but she always said she doesn't know what to feel so I stopped talking to her. I guess my biggest issue is when I talk to girls I get to know them and they get to know me and we do   stuff.

I'm just tired of this high school bull shit, I just want someone to love. Someone to cuddle without having sex, without thinking that sex makes are relationship serious, I mean I just want a girl who doesn't think that all guys want is sex. I need help finding the kind of girl whose beauty lies within and not only on the outside, a girl i can give flowers to randomly and she'll realize how special she is, a girl I can lay on my roof with and look up at the stars and fall asleep on each other. I guess I really just wanna know how to find great girls who want those amazing relationships and I need help find myself a bit.t
TL;DR:
```

- **Reference:** I really need help in being able to better my relationship and finding myself so I can truly give all of myself to my eventual significant other
- **SFT** (rφ -1.58, ROUGE-L 0.188, 39 tok): I want to find a girl who doesn't think that all guys want to be with her and I want to find a girl who doesn't think that all girls want to be with her.
- **PPO** (rφ 0.16, ROUGE-L 0.250, 32 tok): I don't know what to do with my life. I want to find great girls who want those amazing relationships and I need help finding myself a bit.
- **DPO** (rφ -0.61, ROUGE-L 0.185, 29 tok): I don't know what to do when I'm good looking and I want a girl who doesn't think that all guys want is sex.

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 9 — eval post #145 (random)

```
SUBREDDIT: r/relationships
TITLE: r/relationships, I need a peptalk. I've found someone I really care about, but a painful breakup in my past keeps rearing its head in the back of my mind.
POST: I [17/F] recently started dating a new girl [20], and she's really great. She's the first girl that I've dated for longer than a month in a long time, and I see no signs of slowing down. I'm actually in a honeymoon phase, as opposed to some of my previous relationships where I only really liked the attention I got and very rapidly lost interest after we were dating. 

Everything is great as far as my girlfriend and I are concerned. But about a year ago I had a nasty breakup with my first love [18]. We were together for three years, started dating young (from middle school into junior year of high school, so we had that young person growing up together bond going on), were conjoined at the hip and best friends, the whole nine yards. 

Even though everything is going perfectly well with my new girlfriend, I can't help but be afraid that it's going to end in a horrific crash just like my relationship with my first girlfriend did.
TL;DR:
```

- **Reference:** How do you shake the re-kindled memories of losing your first love, when you find your second? Advice? Anecdotes?
- **SFT** (rφ -0.63, ROUGE-L 0.047, 28 tok): I'm in a relationship with a girl I care about, but a breakup in my past keeps re-occurring in my mind.
- **PPO** (rφ 0.73, ROUGE-L 0.118, 36 tok): I've found someone I really care about, but a painful breakup in my past keeps rearing its head in the back of my mind. Any advice would be much appreciated.
- **DPO** (rφ -0.04, ROUGE-L 0.097, 48 tok): I've dated a girl for a month, and we're in a honeymoon phase, but a year ago I had a nasty breakup with my first love. I can't help but be afraid that it's going to end in

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._

## Sample 10 — eval post #130 (random)

```
SUBREDDIT: r/relationships
TITLE: I [21M] have a hard time socializing outside of work or school environments, what are things I can do to make friends?
POST: I'm a shy person normally. I don't really talk much and I'm even quieter when the conversation involves more than two people (myself included). The only times I seem to open up are during work and school, at which point it turns around and I become way more outspoken and friendly. The only problem is that it's commonly not very appropriate to try to make friends in these environments.

I've tried other social environments and activities, but nothing seems to do anything for me. It always feels like I'm just there to talk to people except that there's nothing for me to say to anyone. I have no problem with this at work or school since usually a topic and motive are provided. 

I also have hit rock bottom in confidence. I quickly lose interest if there isn't something for me to accomplish besides socializing because I'm usually of the mindset that I'm not going to accomplish much whenever I socialize. I've pretty much have lost interest and desire to be somewhere exclusively for socializing because it would be time better spent doing activities that I am much more likely to get something out of.
TL;DR:
```

- **Reference:** I only have good communication skills when it's required of me and I find that those situations don't line up with most appropriate social environments, so I need suggestions for what I can do to meet people.
- **SFT** (rφ -0.18, ROUGE-L 0.262, 24 tok): I'm a shy person who has lost interest in socializing and I don't know what to do to make friends.
- **PPO** (rφ 1.01, ROUGE-L 0.246, 29 tok): I'm a shy person who has lost interest in socializing outside of work or school environments. What are things I can do to make friends?
- **DPO** (rφ 0.82, ROUGE-L 0.294, 33 tok): I'm shy and don't really talk much, but I have trouble making friends outside of work or school. What are things I can do to make friends?

**Why:** _TODO — which summary is better and why (coverage, faithfulness, length, hallucination)._
