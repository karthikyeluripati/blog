import Link from "next/link"
import Image from "next/image"
import { ProjectPostMetadata } from "./ProjectPostMetadata"

const ProjectPostPreview = (props: ProjectPostMetadata) => {
    return (
        <div className="">
            <div className="mb-4 p-4 bg-orange-100 border-2 border-black dark:bg-gray-900 dark:border-white">
                {/* Header section */}
                <div className="flex justify-between items-start mb-2">
                    <Link href={`${props.producturl}`}>
                        <h2 className="text-black-900 hover:underline hover:text-orange-700 mr-4">{props.title}</h2>   
                    </Link>
                    <p className="text-sm text-slate-400 whitespace-nowrap">{props.date}</p>
                </div>
                {/* <Link href={`/projectposts/${props.slug}`} >
                    <p className="text-slate-500 ">{props.subtitle}<a className="hover:underline">...read more</a></p>
                </Link> */}

                <p className="text-slate-500 ">{props.subtitle}</p>
 
                {/* Masonry grid with clickable media */}
                {props.media && props.media.length > 0 && (
                    <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-4">
                        {props.media.map((item, index) => (
                            <Link 
                                key={index}
                                href={item.link || '#'} 
                                target={item.link ? "_blank" : undefined}
                                rel={item.link ? "noopener noreferrer" : undefined}
                                className={`${
                                    index === 0 ? 'md:col-span-2 md:row-span-2' : ''
                                } rounded-lg overflow-hidden cursor-pointer`}
                            >
                                {item.type === 'image' ? (
                                    <Image
                                        src={item.url}
                                        alt={`Project media ${index + 1}`}
                                        width={400}
                                        height={300}
                                        className="object-cover w-full h-full hover:opacity-90 transition-opacity"
                                    />
                                ) : (
                                    <video 
                                        className="w-full h-full object-cover hover:opacity-90 transition-opacity"
                                        autoPlay
                                        loop
                                        muted
                                        playsInline
                                        poster={item.thumbnail}
                                    >
                                        <source src={item.url} type="video/mp4" />
                                        Your browser does not support the video tag.
                                    </video>
                                )}
                            </Link>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
 }
 
 export default ProjectPostPreview;