---
title: "Music culture and Computer Generated Music"
date: "2022-12-04"
description: "Discussion of the cultural aspect and how it relates to computer generated music." 
---

> This post is under development and is subject to changes 

For many, music is so much more than just an enjoyable medium. Being a fan of a certain artist or music type can be a way of expressing yourself. Culture also has a big role in what made you a fan in the first place. Music which we can relate to often has a bigger emotional impact. This raises three interesting questions regarding computer generated music (CGM): Can fandom of CGM create an equally strong sense of identity? Can computers generate music which we can relate to? Can emotionless music be good?

## Identity
Can fandom of CGM create an equally strong sense of identity?
The joke template: “How do you know that someone is a <BLANK>? Don't worry, they will tell you” has many forms. Common insertions of <BLANK> is “vegan” for example. The popularity and versatility of this joke give insight into how important a sense of identity is in today's society.

Music taste is one of the major ways people express their personality. The release of Spotify Wrapped is an important time of the year among my friends. I believe that this is an unconscious driving force in many people's choice of music. I can confess that I sometimes actively try to like some music. Subconsciously this might be because I want to be someone that likes that kind of music. Will CGM be able to create that same driving force? Initially,  I believe so. I can totally see that being a fan of CGM as the new hipster thing.

The question becomes more complex when CGM grows to a larger audience. Here I think we can learn a lot by studying the pop industry. Many big pop stars are just the face of an underlying corporation. If people can tie  identity to this, I think they can do the same to CGM. Note that there does not need to be only one AI that creates all the music. Different AIs can be created by different people and optimized to create different types of music. There might be a need for a human face, someone who can perform during concerts and do viral interviews, but this can easily be accounted for.

## #Relatable
Håkan Hällström, is one of Sweden's biggest singers of all time, despite “not being able to sing” according to many. However, it is possible that this is why he is so popular. When his voice breaks, just as yours do when you sing along, it is more relatable.

There are many aspects that decide if you can relate to a song. Legendary music producer Rick Rubin said in [his interview](https://www.youtube.com/watch?v=H_szemxPcTI) on the Lex Fridman podcast that he did not enjoy “west coast music” until he lived on the west coast. We are often more likely to listen to music from our country. From my empirical observations, this is true even if the artist does not sing in that country’s language. Perhaps this is not surprising, if the artist lives in the same area as you, they will have experienced much of the same things, and it is hence easier to relate.

In the same interview, Rick Rubin talked about how johnny cash’s cover of “Hurt” has a completely different meaning when sung by an old person, when regret is expressed by a young person it is tragic, but they have time to fix it, when it is expressed by an old person, it is brutal. There are many more factors that determine if a song relates to us: gender, socioeconomic status, family situation, etc. The list goes on.

So, will we be able to generate music which is relatable? The naive answer is: Yes, just train on data which share the feeling you want to convey. This might work, but it is not obvious that it will. This strategy assumes that there is a way to define this feeling.

## Technical point of view
Imagine a N-dimensional space (N is big) where each dimension corresponds to some property of a song. The songs in our training set will correspond to a point in this space. The song must share some property which puts them close together in at least one of the dimensions. These dimensions are hopefully what is conveying the feeling. If you manage to  gather such data, the strategy stands a chance. However, if the feeling can be expressed in many different ways, meaning that when you pick songs for the training set that you think will convey one concrete feeling, the songs are distributed with high variance in the space. The result will be an average of the points, and this does not necessarily convey the same feeling.

The same argument can be made for many applications of ML. For example, a self-driving car is often trained on data from [both bad and good drivers](https://blog.comma.ai/towards-a-superhuman-driving-agent) (it is too expensive to watch the footage and filter out the bad drivers). Similarly to above, imagene a N-dimensional space where each dimension corresponds to some aspect of driving. Each driver is placed in this space. The hope is that good drivers are clustered around one point in this space, and that bad drivers are deviations from the mean. The average of all bad drivers is in other words a good driver. This is known as an unbiased estimation. Similarly, when picking a set of songs with a concrete feeling, the hope is that most songs are clustered around one point. We also hope that the songs which we mistakenly include are deviations of the mean, and the average of all mistakes is not too far from the ideal point in the space.

There is a lot more to say about the cultural aspect of music and what this will mean for CGM. Here I have barely scratched the surface. I hope to follow up in the future.
