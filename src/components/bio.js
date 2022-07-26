/**
 * Bio component that queries for data
 * with Gatsby's useStaticQuery component
 *
 * See: https://www.gatsbyjs.com/docs/use-static-query/
 */

import * as React from "react"
import { useStaticQuery, graphql } from "gatsby"
import { StaticImage } from "gatsby-plugin-image"

const Bio = () => {
  const data = useStaticQuery(graphql`
    query BioQuery {
      site {
        siteMetadata {
          author {
            name
          }
        }
      }
    }
  `)

  // Set these values by editing "siteMetadata" in gatsby-config.js
  const author = data.site.siteMetadata?.author

  return (
    <div className="bio">
      <StaticImage
        className="bio-avatar"
        layout="fixed"
        formats={["auto", "webp", "avif"]}
        src="../images/profile_pic.jpg"
        width={60}
        height={60}
        quality={100}
        alt="Profile picture"
      />
      <div>
        <p>Written by <strong>{author.name}</strong></p>
        <p>Thoughts, comments or questions? <a href="mailto:lukas.petersson.1999@gmail.com">Reach out!</a></p>
        <p>All code can be found <a href="https://github.com/lukaspetersson/ye_tools">here</a></p>
      </div>
    </div>
  )
}

export default Bio
