import Link from "next/link"
import { ProjectPostMetadata } from "./ProjectPostMetadata"

const ProjectPostPreview = (props:ProjectPostMetadata)=>{
    return (
    // <div className='border border-voilet-300 p-4 rounded-md shadow-md 
    // bg-white'>
    <div className=''>
            <div className="flex mb-4 p-2 bg-orange-100 border-2 border-black dark:bg-gray-900 dark:border-white">
            {/* </div><div className="flex justify-between items-center border-black-solid bg-orange-100 p-1 my-4"> */}
                <Link href={`/projectposts/${props.slug}`} className="flex-grow">
                    <h2 className="text-black-900 hover:underline hover:text-orange-700 mr-4">{props.title}</h2>
                    <p className=" text-slate-500 mt-2">{props.subtitle}</p>
                </Link>
                <p className="text-sm text-slate-400 whitespace-nowrap">{props.date}</p>
            </div>
        </div>
    );
}
export default ProjectPostPreview