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

EDIT_REQUESTS = [
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

DIVERSITY_CONCEPTS = {
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
