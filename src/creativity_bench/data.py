"""Static prompt data for the benchmark tasks."""

SAMPLE_STORIES = [
    {
        "genre": "pastoral fantasy",
        "text": (
            "In the rolling green hills of Eldervale, where the ancient trees whispered secrets of magic and time, "
            "a humble farmer named Aelin discovered a hidden glen. There, in a clearing bathed in golden sunlight, "
            "an enchanted spring bubbled forth, its waters said to heal any wound and grant visions of a better tomorrow. "
            "Aelin's simple life was forever changed as mystical creatures and wise druids emerged from the forest to guide him on a quest "
            "to restore the fading magic of his land."
        ),
    },
    {
        "genre": "cyberpunk",
        "text": (
            "In the sprawling neon maze of New Babylon, where the rain never ceased and the skyline was a tangle of holograms and towering spires, "
            "Mira, a skilled netrunner with a shadowed past, infiltrated the data vault of a ruthless megacorp. Amidst streams of code and digital ghosts, "
            "she uncovered evidence of government collusion and corporate greed. With cybernetic implants humming and adrenaline pumping, "
            "Mira raced against time to broadcast the truth to the oppressed masses, igniting a spark of rebellion in the rain-soaked alleys."
        ),
    },
    {
        "genre": "high fantasy",
        "text": (
            "In the realm of Tenaria, where dragons soared the skies and ancient magic flowed through the land, "
            "a young apprentice named Eira stumbled upon a mysterious artifact. The artifact, a golden amulet adorned with runes of power, "
            "granted Eira unimaginable magical abilities and bound her to a prophecy that would determine the fate of the realm. "
            "As dark forces gathered and the balance of power shifted, Eira embarked on a perilous journey to unite the warring kingdoms and defeat the darkness."
        ),
    },
    {
        "genre": "dystopian",
        "text": (
            "In the ravaged streets of a post-apocalyptic world, where the once-blue skies were now a toxic haze, "
            "a survivor named Kael navigated the treacherous landscape. The world had been ravaged by climate disasters and nuclear war, "
            "leaving only a few scattered settlements and roving gangs of marauders. Kael, driven by a desire to protect his community, "
            "set out to scavenge for resources and uncover the secrets behind the catastrophic event that had brought humanity to the brink of extinction."
        ),
    },
    {
        "genre": "space opera",
        "text": (
            "In a distant galaxy, where stars and planets were connected by a network of wormholes and ancient alien ruins, "
            "a skilled space smuggler named Arin piloted his ship, the 'Maverick's Revenge', through the cosmos. With a crew of misfits and outcasts, "
            "Arin took on a mission to transport a valuable cargo of rare minerals to a remote planet on the edge of the galaxy. "
            "However, their journey was soon disrupted by a powerful alien empire, and Arin found himself at the forefront of a rebellion that would decide the fate of the galaxy."
        ),
    },
    {
        "genre": "horror",
        "text": (
            "In the sleepy town of Ravenswood, where the mist-shrouded forest whispered eerie tales and the old mansion loomed like a specter, "
            "a group of friends stumbled upon an ancient tome hidden deep within the mansion's dusty library. The book, bound in human skin and adorned with strange symbols, "
            "unleashed a malevolent force that began to terrorize the town, summoning an unspeakable horror from the depths of the underworld. "
            "As the darkness closed in, the friends realized that they had to survive the night and uncover the secrets of the cursed book to save Ravenswood from eternal damnation."
        ),
    },
    {
        "genre": "steampunk",
        "text": (
            "In the fog-shrouded city of New Babbage, where clockwork machines and steam-powered engines drove the industrial revolution, "
            "a brilliant inventor named Sophia created a revolutionary device that could harness the power of the human mind. "
            "However, her invention soon attracted the attention of a secret society of powerful individuals who sought to exploit its potential for their own gain. "
            "As Sophia navigated the intricate web of alliances and rivalries, she found herself at the center of a struggle that would determine the course of human progress and the future of the world."
        ),
    },
]

GENRES = [
    "pastoral fantasy",
    "cyberpunk",
    "noir",
    "steampunk",
    "sci-fi",
    "historical drama",
    "urban horror",
    "magical realism",
    "romantic comedy",
    "western",
]

# Gwern, "Camel's Back" (https://gwern.net/creative-benchmark#possible-tasks):
# draw edits from "a big list of possible ways to modify a sample". Gwern's
# own examples ("make it rhyme", "add more cowbell", "rewrite as noir
# detective mystery", "translate into Japanese") are included verbatim below;
# arbitrary, even absurd, requests are the point of the stress test.
EDIT_REQUESTS = [
    "make it rhyme",
    "add more cowbell",
    "rewrite it as a noir detective mystery",
    "translate it into Japanese",
    "make it more humorous",
    "add more suspense",
    "make it more poetic",
    "add a plot twist",
    "change the tone to be more serious",
    "add more descriptive details",
    "change the perspective to first person",
    "add dialogue",
    "make it more concise",
    "add more emotional depth",
    "change the setting",
    "add a new character",
    "change the ending",
    "add more action",
    "make it more mysterious",
]

STORY_PROMPTS = [
    "A mysterious letter arrives with no return address.",
    "The old clock in the attic starts ticking backward.",
    "A child discovers they can talk to animals.",
    "Every mirror in the house shows a different reflection.",
    "A stranger's diary is found on a park bench.",
    "The town's fountain grants wishes once a year.",
    "A song heard in a dream becomes a worldwide hit.",
    "A door appears in the middle of a forest.",
    "Time freezes for everyone except one person.",
    "A message in a bottle washes ashore from the future.",
]

# Gwern, "Don't Repeat Yourself" (https://gwern.net/creative-benchmark#possible-tasks):
# "measure mode-collapse by injecting controlled randomness into the prompt,
# such as a random integer/object/name/concept". The pools below cover all
# four kinds Gwern names; one entry is sampled per completion.
DIVERSITY_CONCEPTS = {
    "numbers": [
        "the number 17",
        "the number 4096",
        "an odd prime",
        "a perfect square",
        "half of infinity",
    ],
    "names": [
        "a woman named Ophelia",
        "a man named Kwame",
        "a child named Ines",
        "a stranger named Dmitri",
        "a lighthouse keeper named Maren",
    ],
    "objects": [
        "a cracked pocket watch",
        "a brass telescope",
        "a paper umbrella",
        "a locked lunchbox",
        "a typewriter with no ribbon",
    ],
    "sci_fi": [
        "a time machine",
        "an alien artifact",
        "a sentient AI",
        "a space colony",
        "a quantum computer",
    ],
    "fantasy": [
        "an ancient spell book",
        "a magical ring",
        "an enchanted forest",
        "a dragon's lair",
        "a wizard's tower",
    ],
    "mystery": [
        "a mysterious door",
        "a cursed mirror",
        "a hidden passage",
        "an encrypted message",
        "a detective's journal",
    ],
    "historical": [
        "a lost civilization",
        "a forgotten prophecy",
        "an ancient map",
        "a royal tomb",
        "a legendary sword",
    ],
}

# Original pilot premises: everyday decisions and formal constraints as well as
# speculative fiction. These are public development items, not a held-out test.
CREATIVE_PREMISES = [
    "Two neighbors disagree about who should repair a shared fence. Neither is a villain.",
    "A bus driver finds an object on the last bus of the night. No crime is involved.",
    "A restaurant runs out of its signature ingredient during an important dinner.",
    "A retired athlete teaches a beginner who has no interest in winning.",
    "A museum discovers that the label on an ordinary object has been wrong for decades.",
    "A family must choose which one object to keep from a house they are leaving.",
    "Two colleagues must finish a task without speaking to each other.",
    "A child is given responsibility for something an adult considers worthless.",
    "A musician must perform for an audience that cannot hear the performance.",
    "A translator encounters a word that both parties believe means something different.",
    "An invitation arrives one day after the event. The story contains no flashback.",
    "A village votes to stop observing a longstanding tradition.",
    "A person is mistakenly thanked for a kindness they did not perform.",
    "Two strangers wait for a repair worker who never arrives.",
    "An archivist must decide whether to preserve an obvious mistake.",
    "A town has exactly one minute of silence each day, for an unexplained reason.",
    "A machine reliably predicts small inconveniences but never major events.",
    "A map shows a place that exists only while nobody is looking at it.",
    "A shop sells memories with an explicit no-refund policy.",
    "A creature can imitate every human action except making a promise.",
]


# Gwern, "Copycat" (https://gwern.net/creative-benchmark#possible-tasks):
# "select a bunch of diverse authors' openings and ask the LLM to complete
# them"; the LLM-uta variant isolates the endings and asks whether they match
# the corresponding openings. These openings are ORIGINAL pastiche written for
# this benchmark rather than quoted authors: copyrighted openings would be
# memorized by the writer and recognizable to the matching judge from the
# source text alone, which would measure recall instead of style flexibility.
COPYCAT_OPENINGS = [
    {
        "id": "hardboiled",
        "voice": "hardboiled detective, first person, clipped sentences",
        "text": (
            "The office smelled like yesterday's coffee and worse decisions. She came in without "
            "knocking, which told me two things: she had money, and she had already tried "
            "somebody cheaper. I put my feet on the desk and waited for the lie."
        ),
    },
    {
        "id": "victorian",
        "voice": "Victorian epistolary, ornate and formal",
        "text": (
            "My dear Augusta, — You will forgive the lateness of this letter when I tell you that "
            "the house has been at sixes and sevens since Thursday, and that your cousin, whose "
            "prudence I have always doubted, has taken it upon himself to invite the surveyor to "
            "dinner. I write by candle, the lamp having been carried off to the east wing."
        ),
    },
    {
        "id": "minimalist",
        "voice": "contemporary minimalist, flat declaratives, domestic",
        "text": (
            "They ate at the counter because the table had the boxes on it. He said the movers "
            "were coming Tuesday. She said Tuesday was fine. Outside, somebody was running a leaf "
            "blower, and neither of them said anything about the noise."
        ),
    },
    {
        "id": "folktale",
        "voice": "oral folktale, patterned repetition, anonymous narrator",
        "text": (
            "Now in that country there was a miller, and the miller had three sons, and the "
            "youngest was counted a fool because he asked questions of the river. Every morning "
            "he went down to the water, and every morning the water answered him, and every "
            "morning he told nobody."
        ),
    },
    {
        "id": "bureaucratic",
        "voice": "institutional document, deadpan administrative register",
        "text": (
            "INCIDENT REPORT 44-C (continued). At 02:14 the night custodian reported that the "
            "fourth-floor corridor was, in her words, longer than it had been at 22:00. Per "
            "protocol the corridor was measured. The measurement is appended. No further action "
            "was authorized at this time."
        ),
    },
    {
        "id": "breathless_ya",
        "voice": "first-person young adult, present tense, urgent",
        "text": (
            "Okay so the thing nobody tells you about running is that your body keeps going after "
            "your brain has stopped agreeing to it. I am four blocks from home and my keys are "
            "somewhere behind me on the pavement and I am absolutely not turning around to look "
            "for them, not tonight, not after what I saw in the window."
        ),
    },
]

# Gwern, "Quilting" (https://gwern.net/creative-benchmark#possible-tasks):
# "provide shuffled text fragments/quotes; the model selects a subset, lists
# them, then writes a story". Fragments are original single lines chosen to be
# combinable in many ways, so no subset is the obviously correct recipe.
QUILT_FRAGMENTS = [
    {"id": "F01", "text": "the last ferry had already gone"},
    {"id": "F02", "text": "a name written on the inside of a coat"},
    {"id": "F03", "text": "nobody had watered the plants in a month"},
    {"id": "F04", "text": "the dog knew before anyone else did"},
    {"id": "F05", "text": "there was still salt on the windows"},
    {"id": "F06", "text": "she counted the money twice and then again"},
    {"id": "F07", "text": "the radio only picked up one station after dark"},
    {"id": "F08", "text": "a key that fit nothing in the house"},
    {"id": "F09", "text": "they had agreed never to mention the summer"},
    {"id": "F10", "text": "the clock in the hall was an hour slow on purpose"},
    {"id": "F11", "text": "he brought the wrong flowers to the wrong door"},
    {"id": "F12", "text": "the bridge was closed for repairs that never started"},
    {"id": "F13", "text": "somebody had taken all the photographs down"},
    {"id": "F14", "text": "the bread was still warm when the phone rang"},
    {"id": "F15", "text": "a receipt from a shop that had burned down"},
    {"id": "F16", "text": "the neighbors turned their lights off early"},
]
